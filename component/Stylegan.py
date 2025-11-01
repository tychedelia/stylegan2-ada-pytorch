"""
StyleGAN2 Extension for TouchDesigner
This extension provides an interface to generate images and animations using NVIDIA's StyleGAN2-ADA model.
"""
import TDFunctions as TDF
import os
import numpy as np
import re
import traceback

class Stylegan:
	"""
	StyleGAN2 implementation for TouchDesigner that enables realtime generation of images with seeds,
	interpolation, and other features from the original StyleGAN2-ADA implementation.
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		
		# Initialize key attributes
		self._G = None  # Generator model
		self._device = None
		self._label = None
		
		# Latent manipulation state
		self._current_w = None  # Store current W vector for manipulation
		self._base_z = None  # Store base Z vector
		self._semantic_directions = {}  # Store semantic direction vectors
		
		# Setup paths on init
		self._setup_paths()
	
	
	def _setup_paths(self):
		"""
		Set up Python paths for StyleGAN2 and virtual environment
		"""
		import sys
		import platform
		
		try:
			# Get the project folder path
			project_path = project.folder
			
			# Path to StyleGAN2 repository
			stylegan_path = r"C:\Users\char\src\github.com\tychedelia\stylegan2-ada-pytorch"
			if not os.path.exists(stylegan_path):
				# Try to find it relative to project path
				possible_path = os.path.join(os.path.dirname(project_path), 'stylegan2-ada-pytorch')
				if os.path.exists(possible_path):
					stylegan_path = possible_path
			
			# Add project and StyleGAN folders to Python path
			paths_to_add = [
				project_path,
				stylegan_path
			]
			
			for path in paths_to_add:
				if os.path.exists(path) and path not in sys.path:
					sys.path.insert(0, path)
			
			# Add torch extension paths
			bias_act_path = os.path.join(stylegan_path, 'torch_extensions', 'bias_act_plugin')
			if os.path.exists(bias_act_path) and bias_act_path not in sys.path:
				sys.path.insert(0, bias_act_path)
			
			upfirdn2d_path = os.path.join(stylegan_path, 'torch_extensions', 'upfirdn2d_plugin')
			if os.path.exists(upfirdn2d_path) and upfirdn2d_path not in sys.path:
				sys.path.insert(0, upfirdn2d_path)
			
			# Set up virtual environment paths
			venv_path = os.path.join(stylegan_path, '.venv')
			if os.path.exists(venv_path):
				if platform.system() == "Windows":
					site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
					scripts_dir = os.path.join(venv_path, 'Scripts')
				else:  # macOS/Linux
					python_dir = next((d for d in os.listdir(os.path.join(venv_path, 'lib')) 
									if d.startswith('python')), None)
					if python_dir:
						site_packages = os.path.join(venv_path, 'lib', python_dir, 'site-packages')
						scripts_dir = os.path.join(venv_path, 'bin')
					else:
						site_packages = None
						scripts_dir = None
				
				if site_packages and os.path.exists(site_packages):
					if site_packages not in sys.path:
						sys.path.insert(0, site_packages)
				
				if scripts_dir and os.path.exists(scripts_dir):
					if scripts_dir not in sys.path:
						sys.path.insert(0, scripts_dir)
						
			print("Paths setup complete")
			
		except Exception as e:
			print(f"Error setting up paths: {e}")
			raise
	
	def SetupNetwork(self):
		"""
		Load the StyleGAN network based on the current parameters
		"""
		try:
			import torch
			import dnnlib
			import legacy
			
			# Get parameters
			network_path = self.ownerComp.par.Network.eval()
			
			# Setup device
			self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			print(f"Setting up network on device: {self._device}")
			
			# Custom size parameters
			size = None
			if self.ownerComp.par.Customsize:
				width = self.ownerComp.par.Width.eval()
				height = self.ownerComp.par.Height.eval()
				size = [height, width]
			
			scale_type = self.ownerComp.par.Scaletype.eval()
			
			# Setup kwargs
			G_kwargs = dnnlib.EasyDict()
			if size is not None:
				G_kwargs.size = size
				G_kwargs.scale_type = scale_type
			
			# Load the network
			print(f'Loading network from "{network_path}"...')
			with dnnlib.util.open_url(network_path) as f:
				self._G = legacy.load_network_pkl(f, custom=(size is not None), **G_kwargs)['G_ema'].to(self._device)
			
			# Setup label
			if self._G.c_dim > 0:
				class_idx = self.ownerComp.par.Class.eval()
				self._label = torch.zeros([1, self._G.c_dim], device=self._device)
				if class_idx >= 0:
					self._label[:, class_idx] = 1
			else:
				self._label = torch.zeros([1, 0], device=self._device)
			
			print(f"Network loaded successfully on {self._device}")
			return True
			
		except Exception as e:
			# Clear state in case of errors
			self._G = None
			self._label = None
			
			print(f"Error setting up network: {e}")
			import traceback
			traceback.print_exc()
			return False
			
	def GenerateFromLatent(self, z=None, w=None, use_style_mixing=False):
		"""
		Generate an image from latent vectors with optional style mixing
		z: Optional Z-space vector to use instead of seed
		w: Optional W-space vector to use directly
		use_style_mixing: Whether to apply style mixing
		"""
		try:
			import torch
			
			# Ensure network is properly set up
			if self._G is None or self._label is None:
				print("Network not set up, attempting to load...")
				if not self.SetupNetwork():
					print("Failed to set up network")
					return None
			
			# Make sure we're using CUDA
			device = torch.device('cuda')
			
			# Get parameters
			truncation_psi = self.ownerComp.par.Truncation.eval()
			noise_mode = self.ownerComp.par.Noisemode.eval()
			space = self.ownerComp.par.Space.eval()
			
			# CRITICAL: Everything must be on the same device
			# Move model to device
			self._G = self._G.to(device)
			
			# Create fresh label tensor on device
			if self._G.c_dim > 0:
				class_idx = self.ownerComp.par.Class.eval()
				self._label = torch.zeros([1, self._G.c_dim], device=device)
				if class_idx >= 0:
					self._label[:, class_idx] = 1
			else:
				self._label = torch.zeros([1, 0], device=device)
			
			# Get or create latent vectors
			if w is not None:
				# Use provided W vector directly
				if not isinstance(w, torch.Tensor):
					w = torch.from_numpy(w).to(device)
				if len(w.shape) == 2:
					w = w.unsqueeze(0)
			elif z is not None:
				# Use provided Z vector
				if not isinstance(z, torch.Tensor):
					z = torch.from_numpy(z).to(device)
				w = self._G.mapping(z, self._label, truncation_psi=truncation_psi)
			else:
				# Generate from seed with manipulations
				seed = self.ownerComp.par.Seed.eval()
				np.random.seed(seed)
				base_z = np.random.randn(1, self._G.z_dim)
				
				# Apply latent space manipulations
				base_z = self._ApplyLatentManipulation(base_z)
				
				z = torch.from_numpy(base_z).to(device)
				w = self._G.mapping(z, self._label, truncation_psi=truncation_psi)
			
			# Store current W for potential manipulation
			self._current_w = w.detach().clone()
			
			# Apply style mixing if requested
			if use_style_mixing:
				w = self._ApplyStyleMixing(w, device, truncation_psi)
			
			# Apply layer-specific modifications
			w = self._ApplyLayerModifications(w, device)

			# Generate image
			with torch.no_grad():
				img = self._G.synthesis(w, noise_mode=noise_mode)
			
			
			# Assume your StyleGAN result is in "img_torch" with shape [N, C, H, W].
			# For TouchDesigner, you usually want [H, W, 4] if it expects RGBA.
			# Also ensure it's uint8 (0..255).
			img_torch = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
			
			# Reorder from [N, C, H, W] to [N, H, W, C].
			img_torch = img_torch.permute(0, 2, 3, 1)
			
			# For a single batch, drop the batch dim -> [H, W, C].
			img_torch = img_torch[0]
			
			# Ensure contiguous memory layout on GPU.
			img_torch = img_torch.contiguous()
			
			if img_torch.shape[2] == 3:
				# Add an alpha channel of 255 on the GPU
				alpha = torch.full(
					(img_torch.shape[0], img_torch.shape[1], 1), 
					255, 
					dtype=torch.uint8, 
					device=img_torch.device
				)
				img_torch = torch.cat([img_torch, alpha], dim=2)
			
			# Now, e.g., shape might be [H, W, 3] or [H, W, 4].
			# If you need an alpha channel, you can cat one here on GPU side, too.
			return img_torch
			
		except Exception as e:
			print(f"Error generating image: {e}")
			import traceback
			traceback.print_exc()
			return None		
			
	def _ApplyLatentManipulation(self, z):
		"""
		Apply real-time manipulations to the latent vector
		"""
		# Get manipulation parameters
		latent_offset_x = self.ownerComp.par.Latentoffsetx.eval() if hasattr(self.ownerComp.par, 'Latentoffsetx') else 0
		latent_offset_y = self.ownerComp.par.Latentoffsety.eval() if hasattr(self.ownerComp.par, 'Latentoffsety') else 0
		latent_offset_z = self.ownerComp.par.Latentoffsetz.eval() if hasattr(self.ownerComp.par, 'Latentoffsetz') else 0
		latent_scale = self.ownerComp.par.Latentscale.eval() if hasattr(self.ownerComp.par, 'Latentscale') else 1.0
		
		# Get dimension selectors
		dim_x = self.ownerComp.par.Latentdimx.eval() if hasattr(self.ownerComp.par, 'Latentdimx') else 0
		dim_y = self.ownerComp.par.Latentdimy.eval() if hasattr(self.ownerComp.par, 'Latentdimy') else 1
		dim_z = self.ownerComp.par.Latentdimz.eval() if hasattr(self.ownerComp.par, 'Latentdimz') else 2
		
		# Apply offsets to selected dimensions
		if latent_offset_x != 0 and dim_x < z.shape[1]:
			z[0, dim_x] += latent_offset_x
		if latent_offset_y != 0 and dim_y < z.shape[1]:
			z[0, dim_y] += latent_offset_y
		if latent_offset_z != 0 and dim_z < z.shape[1]:
			z[0, dim_z] += latent_offset_z
		
		# Apply overall scale
		if latent_scale != 1.0:
			z = z * latent_scale
		
		# Apply semantic directions if any are active
		z = self._ApplySemanticDirections(z)
		
		return z
	
	def _ApplySemanticDirections(self, z):
		"""
		Apply semantic direction vectors from CHOP input to the latent
		"""
		# Check if semantic directions are enabled
		if not (hasattr(self.ownerComp.par, 'Enabledirections') and self.ownerComp.par.Enabledirections.eval()):
			return z
		
		# Get CHOP input for semantic directions
		directions_chop_path = self.ownerComp.par.Semanticdirections.eval() if hasattr(self.ownerComp.par, 'Semanticdirections') else ''
		if not directions_chop_path:
			return z
		
		try:
			directions_chop = op(directions_chop_path)
			if not directions_chop:
				return z
			
			# Get overall scale
			direction_scale = self.ownerComp.par.Directionscale.eval() if hasattr(self.ownerComp.par, 'Directionscale') else 1.0
			
			# Apply directions from CHOP channels
			# Each channel represents a different semantic direction
			# Channel value represents the magnitude for that direction
			for i, channel in enumerate(directions_chop.chans()):
				if i >= z.shape[1]:  # Don't exceed latent dimensions
					break
				
				magnitude = channel.eval() * direction_scale
				if abs(magnitude) > 0.001:  # Only apply if magnitude is significant
					# For now, we apply the magnitude directly to latent dimensions
					# In a full implementation, this would use pre-computed direction vectors
					z[0, i] += magnitude
			
			return z
			
		except Exception as e:
			print(f"Error applying semantic directions: {e}")
			return z
	
	def _ApplyStyleMixing(self, w, device, truncation_psi):
		"""
		Apply style mixing from a secondary seed
		"""
		import torch
		# Get style mixing parameters
		mix_seed = self.ownerComp.par.Stylemixseed.eval() if hasattr(self.ownerComp.par, 'Stylemixseed') else 1
		mix_layers = self.ownerComp.par.Stylemixlayers.eval() if hasattr(self.ownerComp.par, 'Stylemixlayers') else ''
		mix_strength = self.ownerComp.par.Stylemixstrength.eval() if hasattr(self.ownerComp.par, 'Stylemixstrength') else 0
		
		if mix_layers and mix_strength > 0:
			# Generate W from mix seed
			np.random.seed(mix_seed)
			mix_z = torch.from_numpy(np.random.randn(1, self._G.z_dim)).to(device)
			mix_w = self._G.mapping(mix_z, self._label, truncation_psi=truncation_psi)
			
			# Parse layer range (e.g., "0-3" or "4,5,6")
			layer_indices = self._ParseLayerRange(mix_layers)
			
			# Apply mixing
			if layer_indices:
				w_mixed = w.clone()
				for idx in layer_indices:
					if idx < w.shape[1]:
						w_mixed[:, idx] = w[:, idx] * (1 - mix_strength) + mix_w[:, idx] * mix_strength
				return w_mixed
		
		return w
	
	def _ParseLayerRange(self, layer_str):
		"""
		Parse layer range string like '0-3' or '4,5,6' into list of indices
		"""
		indices = []
		try:
			if '-' in layer_str:
				parts = layer_str.split('-')
				if len(parts) == 2:
					start = int(parts[0])
					end = int(parts[1])
					indices = list(range(start, end + 1))
			elif ',' in layer_str:
				indices = [int(x.strip()) for x in layer_str.split(',')]
			else:
				indices = [int(layer_str)]
		except:
			pass
		return indices
	
	def SetSemanticDirectionVectors(self, direction_vectors):
		"""
		Set pre-computed semantic direction vectors for use with CHOP input
		direction_vectors: dict mapping channel names to 512-dimensional numpy arrays
		"""
		self._semantic_directions = direction_vectors
		print(f"Set {len(direction_vectors)} semantic direction vectors")
	
	def _ApplyLayerModifications(self, w, device):
		"""
		Apply layer-specific modifications to W vectors
		"""
		# Get main truncation as fallback
		main_truncation = self.ownerComp.par.Truncation.eval() if hasattr(self.ownerComp.par, 'Truncation') else 0.7
		
		# Check for layer-specific truncation
		coarse_truncation = self.ownerComp.par.Coarsetruncation.eval() if hasattr(self.ownerComp.par, 'Coarsetruncation') else main_truncation
		middle_truncation = self.ownerComp.par.Middletruncation.eval() if hasattr(self.ownerComp.par, 'Middletruncation') else main_truncation
		fine_truncation = self.ownerComp.par.Finetruncation.eval() if hasattr(self.ownerComp.par, 'Finetruncation') else main_truncation
		
		# Only apply if any differ from main truncation
		if coarse_truncation != main_truncation or middle_truncation != main_truncation or fine_truncation != main_truncation:
			w_modified = w.clone()
			w_avg = self._G.mapping.w_avg.unsqueeze(0).unsqueeze(0)
			
			# Apply truncation to different layer groups
			if coarse_truncation is not None and coarse_truncation != 1.0:
				# Coarse layers (0-3) control pose and shape
				for i in range(0, min(4, w.shape[1])):
					w_modified[:, i] = w_avg + (w[:, i] - w_avg) * coarse_truncation
			
			if middle_truncation is not None and middle_truncation != 1.0:
				# Middle layers (4-7) control facial features
				for i in range(4, min(8, w.shape[1])):
					w_modified[:, i] = w_avg + (w[:, i] - w_avg) * middle_truncation
			
			if fine_truncation is not None and fine_truncation != 1.0:
				# Fine layers (8+) control color and texture
				for i in range(8, w.shape[1]):
					w_modified[:, i] = w_avg + (w[:, i] - w_avg) * fine_truncation
			
			return w_modified
		
		return w
	
	def ReloadNetwork(self):
		"""
		Force reload the network
		"""
		# Clear current state
		self._G = None
		self._label = None
		
		# Reload
		return self.SetupNetwork()
	
	def GenerateFromSeed(self):
		"""
		Wrapper for backward compatibility - generates from seed with manipulations
		"""
		return self.GenerateFromLatent()
	
	def CookOutput(self):
		"""
		Generate an image based on current parameters
		Returns a torch tensor with the generated image
		"""
		# Check if style mixing is enabled
		use_style_mixing = False
		if hasattr(self.ownerComp.par, 'Enablestylemixing'):
			use_style_mixing = self.ownerComp.par.Enablestylemixing.eval()
		
		return self.GenerateFromLatent(use_style_mixing=use_style_mixing)