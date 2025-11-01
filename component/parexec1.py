# me - this DAT
# par - the Par object that has changed
# val - the current value
# prev - the previous value
# 
# Make sure the corresponding toggle is enabled in the Parameter Execute DAT.

def onValueChange(par, prev):
    # Handle value parameter changes if needed
    return

def onPulse(par):
    # Get the parent component and extension
    ext = parent.stylegan
    
    if par.name == 'Reload':
        # Reload the StyleGAN network
        print("Reloading StyleGAN network...")
        ext.SetupNetwork()
    
    elif par.name == 'Randomseed':
        # Generate a random seed
        import random
        new_seed = random.randint(0, 1000000)
        parent().par.Seed = new_seed
        print(f"New random seed: {new_seed}")
    
    elif par.name == 'Resetdefaults':
        # Reset all parameters to defaults
        print("Resetting parameters to defaults...")
        # Import and call the reset function from execute1
        exec_module = op('execute1').module
        if hasattr(exec_module, 'resetToDefaults'):
            exec_module.resetToDefaults()
        else:
            print("Reset function not found")
    
    return

def onValuesChanged(changes):
    # Handle bulk parameter changes if needed
    return

# Other callback handlers are kept empty for now
def onExpressionChange(par, val, prev):
    return

def onExportChange(par, val, prev):
    return

def onEnableChange(par, val, prev):
    return

def onModeChange(par, val, prev):
    return