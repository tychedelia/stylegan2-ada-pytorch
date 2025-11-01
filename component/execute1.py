def onCreate():
	import os
	import sys
	import platform
	
	# Get the project folder path
	project_path = project.folder
	print(f"Project folder: {project_path}")
	
	# Path to StyleGAN2 repository
	stylegan_path = r"C:\Users\char\src\github.com\tychedelia\stylegan2-ada-pytorch"
	if not os.path.exists(stylegan_path):
		# Try to find it relative to project path
		possible_path = os.path.join(os.path.dirname(project_path), 'stylegan2-ada-pytorch')
		if os.path.exists(possible_path):
			stylegan_path = possible_path
	
	print(f"StyleGAN2 path: {stylegan_path}")
	
	# Add project and StyleGAN folders to Python path
	paths_to_add = [
		project_path,
		stylegan_path
	]
	
	for path in paths_to_add:
		if os.path.exists(path) and path not in sys.path:
			print(f"Adding to Python path: {path}")
			sys.path.insert(0, path)
	
	bias_act_path = os.path.join(stylegan_path, 'torch_extensions', 'bias_act_plugin')
	if os.path.exists(bias_act_path) and bias_act_path not in sys.path:
		sys.path.insert(0, bias_act_path)
	
	upfirdn2d_path = os.path.join(stylegan_path, 'torch_extensions', 'upfirdn2d_plugin')
	if os.path.exists(upfirdn2d_path) and upfirdn2d_path not in sys.path:
		sys.path.insert(0, upfirdn2d_path)
	
	# Set up virtual environment paths
	venv_path = os.path.join(stylegan_path, '.venv')
	if os.path.exists(venv_path):
		print(f"Found virtual environment at: {venv_path}")
		
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
				print(f"Adding virtual environment site-packages: {site_packages}")
				sys.path.insert(0, site_packages)
		
		if scripts_dir and os.path.exists(scripts_dir):
			if scripts_dir not in sys.path:
				print(f"Adding virtual environment scripts: {scripts_dir}")
				sys.path.insert(0, scripts_dir)
	return

def resetToDefaults():
	"""
	Reset all parameters to their default values
	Call this function to reset the component to default state
	"""
	comp = op('..')
	
	# Reset all parameters to defaults if they exist
	try:
		if hasattr(comp.par, 'Width'): comp.par.Width = 1024
		if hasattr(comp.par, 'Height'): comp.par.Height = 1024
		if hasattr(comp.par, 'Seed'): comp.par.Seed = 0
		if hasattr(comp.par, 'Truncation'): comp.par.Truncation = 0.7
		if hasattr(comp.par, 'Customsize'): comp.par.Customsize = False
		if hasattr(comp.par, 'Class'): comp.par.Class = -1
		if hasattr(comp.par, 'Noisemode'): comp.par.Noisemode = 'const'
		if hasattr(comp.par, 'Space'): comp.par.Space = 'z'
		if hasattr(comp.par, 'Scaletype'): comp.par.Scaletype = 'pad'
		
		# Latent controls
		if hasattr(comp.par, 'Latentoffsetx'): comp.par.Latentoffsetx = 0.0
		if hasattr(comp.par, 'Latentoffsety'): comp.par.Latentoffsety = 0.0
		if hasattr(comp.par, 'Latentoffsetz'): comp.par.Latentoffsetz = 0.0
		if hasattr(comp.par, 'Latentscale'): comp.par.Latentscale = 1.0
		if hasattr(comp.par, 'Coarsetruncation'): comp.par.Coarsetruncation = 0.7
		if hasattr(comp.par, 'Middletruncation'): comp.par.Middletruncation = 0.7
		if hasattr(comp.par, 'Finetruncation'): comp.par.Finetruncation = 0.7
		
		# Style mixing
		if hasattr(comp.par, 'Enablestylemixing'): comp.par.Enablestylemixing = False
		if hasattr(comp.par, 'Stylemixseed'): comp.par.Stylemixseed = 1
		if hasattr(comp.par, 'Stylemixlayers'): comp.par.Stylemixlayers = '4-7'
		if hasattr(comp.par, 'Stylemixstrength'): comp.par.Stylemixstrength = 0.0
		
		# Semantic directions
		if hasattr(comp.par, 'Semanticdirections'): comp.par.Semanticdirections = ''
		if hasattr(comp.par, 'Directionscale'): comp.par.Directionscale = 1.0
		if hasattr(comp.par, 'Enabledirections'): comp.par.Enabledirections = False
		
		# Advanced
		if hasattr(comp.par, 'Allowrecompile'): comp.par.Allowrecompile = False
		
		print("Reset all parameters to defaults")
		
	except Exception as e:
		print(f"Error resetting defaults: {e}")

def recreateParameters(comp):
	"""
	Force recreation of all parameters with current defaults
	Use this if you need to completely rebuild the parameter interface
	"""
	# Clear all existing custom pages
	for page in comp.customPages:
		page.destroy()
	
	# Force recreation by calling onCreate again

	# Clear any existing custom pages first
	for page in comp.customPages:
		page.destroy()
	
	# Create parameter pages
	setup_page = comp.appendCustomPage('Setup')
	generation_page = comp.appendCustomPage('Generation')
	latent_page = comp.appendCustomPage('Latent Control')
	style_mixing_page = comp.appendCustomPage('Style Mixing')
	semantic_page = comp.appendCustomPage('Semantic Directions')
	advanced_page = comp.appendCustomPage('Advanced')
	
	# Setup Page Parameters
	p = setup_page.appendFile('Network', label='Network Path')[0]
	p.default = ''
	
	p = setup_page.appendToggle('Customsize', label='Use Custom Size')[0]
	p.default = False
	
	p = setup_page.appendInt('Width', label='Width')[0]
	p.default = 1024
	p.min = 64
	p.max = 2048
	p.clampMin = True
	p.clampMax = True
	p.normMin = 64
	p.normMax = 2048
	
	p = setup_page.appendInt('Height', label='Height')[0]
	p.default = 1024
	p.min = 64
	p.max = 2048
	p.clampMin = True
	p.clampMax = True
	p.normMin = 64
	p.normMax = 2048
	
	p = setup_page.appendMenu('Scaletype', label='Scale Type')[0]
	p.menuNames = ['pad', 'padside', 'symm', 'symmside']
	p.menuLabels = ['Pad', 'Pad Side', 'Symmetric', 'Symmetric Side']
	p.default = 'pad'
	
	p = setup_page.appendInt('Class', label='Class Index')[0]
	p.default = -1
	p.min = -1
	p.max = 1000
	p.clampMin = True
	p.clampMax = True
	p.normMin = -1
	p.normMax = 1000
	
	# Generation Page Parameters
	p = generation_page.appendInt('Seed', label='Seed')[0]
	p.default = 0
	p.min = 0
	p.max = 1000000  # Reasonable range for seeds
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0
	p.normMax = 1000000
	
	p = generation_page.appendFloat('Truncation', label='Truncation Psi')[0]
	p.default = 0.7  # 0.7 is typically a good default for quality
	p.min = 0.0
	p.max = 2.0
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0.0
	p.normMax = 2.0
	
	p = generation_page.appendMenu('Noisemode', label='Noise Mode')[0]
	p.menuNames = ['const', 'random', 'none']
	p.menuLabels = ['Constant', 'Random', 'None']
	p.default = 'const'
	
	p = generation_page.appendMenu('Space', label='Latent Space')[0]
	p.menuNames = ['z', 'w']
	p.menuLabels = ['Z Space', 'W Space']
	p.default = 'z'
	
	# Latent Control Page Parameters
	p = latent_page.appendFloat('Latentoffsetx', label='Latent Offset X')[0]
	p.default = 0.0
	p.min = -512.0  # Maximum range for extreme abstract exploration
	p.max = 512.0
	p.clampMin = True
	p.clampMax = True
	p.normMin = -512.0
	p.normMax = 512.0
	
	p = latent_page.appendFloat('Latentoffsety', label='Latent Offset Y')[0]
	p.default = 0.0
	p.min = -512.0
	p.max = 512.0
	p.clampMin = True
	p.clampMax = True
	p.normMin = -512.0
	p.normMax = 512.0
	
	p = latent_page.appendFloat('Latentoffsetz', label='Latent Offset Z')[0]
	p.default = 0.0
	p.min = -512.0
	p.max = 512.0
	p.clampMin = True
	p.clampMax = True
	p.normMin = -512.0
	p.normMax = 512.0
	
	# Add dimension selectors
	p = latent_page.appendInt('Latentdimx', label='X Controls Dimension')[0]
	p.default = 0
	p.min = 0
	p.max = 511
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0
	p.normMax = 511
	
	p = latent_page.appendInt('Latentdimy', label='Y Controls Dimension')[0]
	p.default = 1
	p.min = 0
	p.max = 511
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0
	p.normMax = 511
	
	p = latent_page.appendInt('Latentdimz', label='Z Controls Dimension')[0]
	p.default = 2
	p.min = 0
	p.max = 511
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0
	p.normMax = 511
	
	p = latent_page.appendFloat('Latentscale', label='Latent Scale')[0]
	p.default = 1.0
	p.min = 0.1  # Prevent zero scale
	p.max = 1.5  # Prevent extreme scaling
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0.1
	p.normMax = 1.5
	
	# Layer-specific truncation controls
	p = latent_page.appendFloat('Coarsetruncation', label='Coarse Layers (0-3) Truncation')[0]
	p.default = 0.7  # Match main truncation default
	p.min = 0.0
	p.max = 2.0
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0.0
	p.normMax = 2.0
	
	p = latent_page.appendFloat('Middletruncation', label='Middle Layers (4-7) Truncation')[0]
	p.default = 0.7
	p.min = 0.0
	p.max = 2.0
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0.0
	p.normMax = 2.0
	
	p = latent_page.appendFloat('Finetruncation', label='Fine Layers (8+) Truncation')[0]
	p.default = 0.7
	p.min = 0.0
	p.max = 2.0
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0.0
	p.normMax = 2.0
	
	# Style Mixing Page Parameters
	p = style_mixing_page.appendToggle('Enablestylemixing', label='Enable Style Mixing')[0]
	p.default = False
	
	p = style_mixing_page.appendInt('Stylemixseed', label='Mix Source Seed')[0]
	p.default = 1  # Default to different seed than main
	p.min = 0
	p.max = 1000000  # Reasonable range for seeds
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0
	p.normMax = 1000000
	
	p = style_mixing_page.appendStr('Stylemixlayers', label='Mix Layers (e.g. 0-3 or 4,5,6)')[0]
	p.default = '4-7'  # Middle layers are often good for mixing
	
	p = style_mixing_page.appendFloat('Stylemixstrength', label='Mix Strength')[0]
	p.default = 0.0  # Start with no mixing
	p.min = 0.0
	p.max = 1.0
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0.0
	p.normMax = 1.0
	
	# Semantic Directions Page Parameters
	p = semantic_page.appendCHOP('Semanticdirections', label='Semantic Directions CHOP')[0]
	p.default = ''
	
	p = semantic_page.appendFloat('Directionscale', label='Overall Direction Scale')[0]
	p.default = 1.0
	p.min = 0.0
	p.max = 3.0
	p.clampMin = True
	p.clampMax = True
	p.normMin = 0.0
	p.normMax = 3.0
	
	p = semantic_page.appendToggle('Enabledirections', label='Enable Semantic Directions')[0]
	p.default = False
	
	# Advanced Page Parameters
	p = advanced_page.appendToggle('Allowrecompile', label='Allow CUDA Recompile')[0]
	p.default = False
	
	# Create buttons
	p = setup_page.appendPulse('Reload', label='Reload Network')[0]
	p = generation_page.appendPulse('Randomseed', label='Random Seed')[0]
	p = setup_page.appendPulse('Resetdefaults', label='Reset to Defaults')[0]

