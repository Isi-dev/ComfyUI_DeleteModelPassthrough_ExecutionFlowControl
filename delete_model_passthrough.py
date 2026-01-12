import torch, gc, psutil
import comfy.model_management as mm
from comfy.model_management import loaded_models, free_memory, get_torch_device
from comfy.comfy_types import IO
from nodes import ControlNetLoader, VAELoader, UNETLoader, CLIPTextEncode
from comfy_extras.nodes_model_patch import ModelPatchLoader

try:
    from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
    GGUF_AVAILABLE = True
except ImportError:
    print("⚠️ ComfyUI_GGUF not available - ControlledUnetLoaderGGUF will not work")
    GGUF_AVAILABLE = False
    UnetLoaderGGUF = None

class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False
any_typ = AnyType("*")


def hard_free_model(model):
    if model is None:
        return
        
    try:
        model_type = type(model).__name__
        print(f"Freeing {model_type} model...")
        
        # Handle dictionary-style models (common in some workflows)
        if isinstance(model, dict):
            for key, value in list(model.items()):
                if hasattr(value, 'parameters') or hasattr(value, 'model'):
                    hard_free_model(value)
                del model[key]
            return
        
        # Handle ModelPatcher objects (common in ComfyUI)
        if hasattr(model, 'model') and model.model is not None:
            inner_model = model.model
            if hasattr(inner_model, "parameters"):
                for p in inner_model.parameters():
                    if p is not None:
                        p.detach_()
                        del p
            model.model = None
        
        # CLIP-specific cleanup - some CLIP models have additional attributes
        if hasattr(model, "transformer") and model.transformer is not None:
            if hasattr(model.transformer, "parameters"):
                for p in model.transformer.parameters():
                    if p is not None:
                        p.detach_()
                        del p
            model.transformer = None
            
        # Handle tokenizer if present (some CLIP implementations)
        if hasattr(model, "tokenizer"):
            model.tokenizer = None
            
        # Standard parameter cleanup
        if hasattr(model, "parameters"):
            for p in model.parameters():
                if p is not None:
                    p.detach_()
                    del p
        
        # Clear buffers and tensor attributes
        if hasattr(model, "buffers"):
            for b in model.buffers():
                if b is not None:
                    b.detach_()
                    del b
        
        # Clear any tensor attributes
        for attr_name in list(vars(model).keys()):
            attr = getattr(model, attr_name)
            if isinstance(attr, torch.Tensor):
                attr.detach_()
                delattr(model, attr_name)
                
    except Exception as e:
        print(f"❌ Error during model freeing: {e}")

def identify_model_type(model_obj):
    """Identify what type of model this is"""
    if model_obj is None:
        return "Unknown"
    
    if isinstance(model_obj, dict):
        model_types = []
        for key, value in model_obj.items():
            if hasattr(value, '__class__'):
                model_types.append(f"{key}:{value.__class__.__name__}")
        return f"DictContainer[{', '.join(model_types)}]"
    
    cls_name = model_obj.__class__.__name__.lower()
    
    if 'clip' in cls_name:
        return "CLIP"
    elif 'unet' in cls_name:
        return "UNet"
    elif 'vae' in cls_name:
        return "VAE"
    elif 'control' in cls_name:
        return "ControlNet"
    elif 'modelpatcher' in cls_name:
        return "ModelPatcher"
    elif 'diffusion' in cls_name:
        return "DiffusionModel"
    else:
        return f"Unknown ({model_obj.__class__.__name__})"


def print_currently_loaded():
    """Print models currently loaded in ComfyUI's management system"""
    current_models = mm.loaded_models()
    print("📋 Models in loaded_models():")
    if not current_models:
        print("   No models currently managed")
        return 0
    
    for i, model in enumerate(current_models):
        try:
            model_type = identify_model_type(model)
            # Remove the problematic formatting entirely
            if hasattr(model, 'model_memory'):
                memory_used = model.model_memory()
                print(f"   {i}: {model_type} ({memory_used})")
            else:
                print(f"   {i}: {model_type} (unknown memory)")
        except Exception as e:
            print(f"   {i}: [Error: {e}]")
    
    return len(current_models)

class DeleteModelPassthrough:
    """
    ComfyUI Custom Node: Properly deletes models using ComfyUI's memory management system
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"data": (any_typ,), "model": (any_typ,)}}
 
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "Memory Management"

    def run(self, data, model):
        if model is None:
            print("⚠️ No model provided to delete")
            return (data,)
            
        # Get memory stats before deletion
        before_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        before_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        before_ram = psutil.virtual_memory().percent


        
        model_type = identify_model_type(model)
        # print(f"Target model type: {model_type}")
        
        # Print current state
        initial_count = print_currently_loaded()
        
        # Try to remove from ComfyUI's management system
        model_removed = False
        current_models = mm.loaded_models()
        
        if model in current_models:
            # print("🗑Removing model from ComfyUI management...")
            current_models.remove(model)
            model_removed = True
        else:
            # Check if it's a ModelPatcher or wrapped model
            for managed_model in current_models:
                try:
                    if (hasattr(managed_model, 'model') and 
                        (managed_model.model is model or 
                         (hasattr(model, 'model') and managed_model.model is model.model))):
                        # print("🗑Removing wrapped model from ComfyUI management...")
                        current_models.remove(managed_model)
                        model_removed = True
                        break
                except:
                    continue
        
        # Free memory using ComfyUI's proper methods
        # print("Freeing memory using ComfyUI's system...")
        mm.free_memory(1e30, mm.get_torch_device(), mm.loaded_models())
        
        # Additional forceful cleanup
        # print("Forceful cleanup...")
        hard_free_model(model)

        try:
            del model
        except:
            pass

        
        # ComfyUI's cache cleanup
        mm.soft_empty_cache(force=True)
        
        # Standard Python cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Get memory stats after deletion
        after_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        after_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        after_ram = psutil.virtual_memory().percent

        # Print final state
        final_count = print_currently_loaded()
        
        print(f"Managed models: {initial_count} → {final_count}")
        print(f"System RAM change: {before_ram - after_ram:+.2f}%")
        
        if torch.cuda.is_available():
            vram_freed = (before_vram - after_vram) / (1024 * 1024 * 1024)
            reserved_freed = (before_reserved - after_reserved) / (1024 * 1024 * 1024)
            print(f"GPU allocated freed: {vram_freed:.3f} GB")
            print(f"GPU reserved freed: {reserved_freed:.3f} GB")
            # print(f"Final allocated: {after_vram / (1024 * 1024 * 1024):.3f} GB")
            # print(f"Final reserved: {after_reserved / (1024 * 1024 * 1024):.3f} GB")
        
        print("=" * 60)
        
        # Success determination
        if model_removed:
            print("SUCCESS: Model removed from management system!")
        elif torch.cuda.is_available() and reserved_freed > 0.1:
            print("SUCCESS: Significant memory freed (model may use custom management)")
        else:
            print("⚠PARTIAL: Model may still be referenced somewhere")
        
        print("=" * 60)

        return (data,)


def safe_hard_free_model(model):
    """Memory-safe model freeing without expensive operations"""
    if model is None:
        return
        
    try:
        # Minimal identification without memory overhead
        model_type = model.__class__.__name__
        
        # Handle dictionary-style models - but do it safely
        if isinstance(model, dict):
            # Clear dictionary without iterating through values
            model.clear()
            return
        
        # Quick and dirty cleanup - avoid iterating through parameters if possible
        if hasattr(model, 'model') and model.model is not None:
            # Just nullify the reference instead of iterating parameters
            model.model = None
        
        # Nullify common heavy attributes without iteration
        for attr_name in ['transformer', 'tokenizer', 'encoder', 'decoder']:
            if hasattr(model, attr_name):
                setattr(model, attr_name, None)
                
        # If we must detach, do it in chunks with garbage collection
        if hasattr(model, "parameters"):
            param_count = 0
            for p in model.parameters():
                if p is not None:
                    try:
                        p.detach_()
                    except:
                        pass
                    param_count += 1
                    # Periodic garbage collection to prevent memory buildup
                    if param_count % 1000 == 0:
                        gc.collect()
        
        # Clear the object itself
        try:
            del model
        except:
            pass
            
    except Exception as e:
        # Silent error handling to avoid memory overhead
        pass

def safe_print_loaded_models():
    """Memory-safe way to print loaded models"""
    try:
        current_models = mm.loaded_models()
        if not current_models:
            print("No models currently managed")
            return 0
        
        print(f"Managed models count: {len(current_models)}")
        # Just show count, avoid accessing model properties
        return len(current_models)
        
    except:
        return 0

def quick_identify_model(model_obj):
    """Quick identification without memory overhead"""
    if model_obj is None:
        return "Unknown"
    return model_obj.__class__.__name__



class DeleteModelPassthroughLight:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"data": (any_typ,), "model": (any_typ,)}}
 
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "Memory Management"

    def run(self, data, model):
        if model is None:
            return (data,)
            
        # Get memory stats BEFORE any operations
        before_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        before_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        
        # Quick identification
        model_type = quick_identify_model(model)
        
        # Safe model counting
        initial_count = safe_print_loaded_models()
        
        # Remove ONLY the target model from management
        model_removed = False
        try:
            current_models = mm.loaded_models()
            if model in current_models:
                current_models.remove(model)
                model_removed = True
                print(f"Removed target model from tracking")
        except Exception as e:
            print(f"Error removing from tracking: {e}")
    
        # Don't use free_memory() with huge values as it will delete other models
        
        # Instead, manually free only our target model
        safe_hard_free_model(model)
        
        # Optional: Use free_memory() correctly if we know how much this model used
        # But better to avoid it entirely for single-model deletion
        
        # Gentle cleanup without affecting other models
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Get memory stats after deletion
        after_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        after_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        
        # Safe final count
        final_count = safe_print_loaded_models()
        
        # Check if other models were affected
        other_models_affected = initial_count - final_count > 1 if model_removed else initial_count - final_count > 0
        
        if torch.cuda.is_available():
            vram_freed = (before_vram - after_vram) / (1024 * 1024 * 1024)
            reserved_freed = (before_reserved - after_reserved) / (1024 * 1024 * 1024)
            print(f"Freed: {reserved_freed:.3f}GB | Models: {initial_count}→{final_count}")
            
            if other_models_affected:
                print(f"⚠️ WARNING: {initial_count - final_count} models were affected, not just the target!")
        
        return (data,)

class SmartClipDeleter:
    """
    Automatically detects if CLIP is in RAM or VRAM and handles appropriately
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"data": (any_typ,), "clip_model": ("CLIP",)}}
 
    RETURN_TYPES = (any_typ,)
    FUNCTION = "delete_smart"
    CATEGORY = "Memory Management"

    def delete_smart(self, data, clip_model):
        if clip_model is None:
            return (data,)
            
        # Detect where the model is loaded
        model_in_vram = False
        model_in_ram = False
        
        # Check if model has parameters on GPU
        if hasattr(clip_model, "parameters"):
            for p in clip_model.parameters():
                if p is not None and p.device.type != 'cpu':
                    model_in_vram = True
                    break
                elif p is not None:
                    model_in_ram = True
        
        print(f"📍 CLIP model location: {'VRAM' if model_in_vram else 'RAM'}")
        
        if model_in_vram:
            # Use VRAM cleanup approach
            try:
                if hasattr(clip_model, 'model_memory'):
                    model_size = clip_model.model_memory()
                    mm.free_memory(model_size, mm.get_torch_device())
                else:
                    mm.soft_empty_cache()
            except:
                pass
                
        else:
            # Use RAM cleanup approach
            try:
                # Clear model attributes
                for attr in ['transformer', 'tokenizer', 'text_model', 'visual', 'model']:
                    if hasattr(clip_model, attr):
                        setattr(clip_model, attr, None)
                
                # Force garbage collection
                gc.collect()
                gc.collect()
                
            except:
                pass
    
        return (data,)

class ControlledControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Get the original input types and add trigger
        original_types = ControlNetLoader.INPUT_TYPES()
        if "required" in original_types:
            original_types["required"]["trigger"] = (any_typ, {"default": None})
        else:
            original_types["required"] = {"trigger": (any_typ, {"default": None})}
        return original_types

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Memory Management"
    TITLE = "Controlled ControlNet Loader"

    def load_controlnet(self, trigger, *args, **kwargs):
        if trigger is None:
            print("⏸️  ControlNet loading paused - no trigger received")
            return (None,)
        
        print(f"🚀 Loading ControlNet...")
        # Simply call the original class method
        return ControlNetLoader.load_controlnet(self, *args, **kwargs)


class ControlledVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Copy INPUT_TYPES from VAELoader and extend it
        base = VAELoader.INPUT_TYPES(cls)
        base = base.copy()
        base["required"] = dict(base.get("required", {}))

        # Add execution trigger
        base["required"]["trigger"] = (any_typ, {"default": None})

        return base

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "Memory Management"
    TITLE = "Controlled VAE Loader"

    def load_vae(self, vae_name, trigger=None):
        # Delay execution if trigger is not provided
        if trigger is None:
            print("VAE loading paused - no trigger received")
            return (None,)

        print(" Loading VAE...")
        loader = VAELoader()
        return loader.load_vae(vae_name)



class ControlledUnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        if not GGUF_AVAILABLE:
            return {"required": {"trigger": (any_typ, {"default": None})}}
        
        # Get the original input types and add trigger
        original_types = UnetLoaderGGUF.INPUT_TYPES()
        if "required" in original_types:
            original_types["required"]["trigger"] = (any_typ, {"default": None})
        else:
            original_types["required"] = {"trigger": (any_typ, {"default": None})}
        return original_types

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "Memory Management"
    TITLE = "Controlled UNet Loader (GGUF)"

    def load_unet(self, trigger, *args, **kwargs):
        if not GGUF_AVAILABLE:
            print("❌ ComfyUI_GGUF not installed - cannot load UNet")
            return (None,)
            
        if trigger is None:
            print("⏸UNet loading paused - no trigger received")
            return (None,)
        
        print(f"Loading UNet...")
        # Simply call the original class method
        return UnetLoaderGGUF.load_unet(self, *args, **kwargs)


class ControlledModelPatchLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Get the original input types and add trigger
        original_types = ModelPatchLoader.INPUT_TYPES()
        if "required" in original_types:
            original_types["required"]["trigger"] = (any_typ, {"default": None})
        else:
            original_types["required"] = {"trigger": (any_typ, {"default": None})}
        return original_types

    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "load_model_patch"
    CATEGORY = "Memory Management"
    TITLE = "Controlled Model Patch Loader"
    EXPERIMENTAL = True

    def load_model_patch(self, trigger, *args, **kwargs):
        if trigger is None:
            print("⏸️  Model Patch loading paused - no trigger received")
            return (None,)
        
        print(f"🚀 Loading Model Patch...")
        # Simply call the original class method
        return ModelPatchLoader.load_model_patch(self, *args, **kwargs)

class ControlledUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Get original input types and add trigger
        original_types = UNETLoader.INPUT_TYPES()

        if "required" in original_types:
            original_types["required"]["trigger"] = (any_typ, {"default": None})
        else:
            original_types["required"] = {"trigger": (any_typ, {"default": None})}

        return original_types

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "Memory Management"
    TITLE = "Controlled UNet Loader"

    def load_unet(self, trigger, *args, **kwargs):
        # Pause loading if no trigger received
        if trigger is None:
            print("⏸️  UNet loading paused — no trigger received")
            return (None,)

        print("🚀 Loading UNet...")

        # Forward call to original loader
        return UNETLoader.load_unet(self, *args, **kwargs)

class ControlledCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        # Get original input types and add trigger
        original_types = CLIPTextEncode.INPUT_TYPES()

        if "required" in original_types:
            original_types["required"]["trigger"] = (any_typ, {"default": None})
        else:
            original_types["required"] = {"trigger": (any_typ, {"default": None})}

        return original_types

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "encode"
    CATEGORY = "Memory Management"
    TITLE = "Controlled CLIP Text Encode"
    DESCRIPTION = "A trigger-controlled wrapper around CLIPTextEncode."

    def encode(self, trigger, *args, **kwargs):
        # Pause encoding if no trigger value given
        if trigger is None:
            print("⏸️  CLIP text encoding paused — no trigger received")
            return (None,)

        print("🚀 Encoding text with CLIP...")

        # Forward to original class method
        return CLIPTextEncode.encode(self, *args, **kwargs)


# Experimental Factory code
# def make_trigger_controlled_node(original_class, title_suffix=" (Controlled)", category="Memory Management"):
#     """
#     Creates a trigger-controlled wrapper class around any ComfyUI node class.
#     """

#     original_function_name = getattr(original_class, "FUNCTION", None)
#     if original_function_name is None:
#         raise ValueError(f"{original_class.__name__} does not define FUNCTION")

#     # --------------------------
#     # 1. Build INPUT_TYPES wrapper
#     # --------------------------
#     @classmethod
#     def INPUT_TYPES(cls):
#         original_types = original_class.INPUT_TYPES()

#         # Insert trigger field
#         if "required" in original_types:
#             original_types["required"]["trigger"] = (any_typ, {"default": None})
#         else:
#             original_types["required"] = {"trigger": (any_typ, {"default": None})}

#         return original_types

#     # --------------------------
#     # 2. Wrapped FUNCTION method
#     # --------------------------
#     def wrapped_function(self, trigger, *args, **kwargs):
#         if trigger is None:
#             print(f"⏸️  {original_class.__name__} paused — no trigger received")
#             return (None,)
        
#         print(f"🚀 Executing {original_class.__name__}...")
#         original_fn = getattr(original_class, original_function_name)
#         return original_fn(self, *args, **kwargs)

#     # --------------------------
#     # 3. Build dynamic class attrs
#     # --------------------------
#     attrs = {
#         "INPUT_TYPES": INPUT_TYPES,
#         "FUNCTION": original_function_name,
#         original_function_name: wrapped_function,
#         "RETURN_TYPES": getattr(original_class, "RETURN_TYPES", ()),
#         "CATEGORY": category,
#         "TITLE": getattr(original_class, "TITLE", original_class.__name__ + title_suffix),
#     }

#     # Copy DESCRIPTION if present
#     if hasattr(original_class, "DESCRIPTION"):
#         attrs["DESCRIPTION"] = getattr(original_class, "DESCRIPTION")

#     # --------------------------
#     # 4. Create new class
#     # --------------------------
#     new_class_name = f"Controlled_{original_class.__name__}"

#     return type(new_class_name, (original_class,), attrs)
    
# How to use 'Experimental Factory code':
# ControlledUNETLoader = make_trigger_controlled_node(UNETLoader)


NODE_CLASS_MAPPINGS = {
    "DeleteModelPassthroughLight": DeleteModelPassthroughLight,
    "DeleteModelPassthrough": DeleteModelPassthrough,
    "ControlledUNETLoader": ControlledUNETLoader,
    "ControlledUnetLoaderGGUF": ControlledUnetLoaderGGUF,
    "ControlledControlNetLoader": ControlledControlNetLoader,
    "ControlledVAELoader": ControlledVAELoader,
    "ControlledCLIPTextEncode": ControlledCLIPTextEncode,
    "ControlledModelPatchLoader": ControlledModelPatchLoader,
    "SmartClipDeleter": SmartClipDeleter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeleteModelPassthroughLight": "Delete Model Light (Passthrough Any)",
    "DeleteModelPassthrough": "Delete Model (Passthrough Any)",
    "ControlledUNETLoader": "Controlled Load Diffusion Model",
    "ControlledUnetLoaderGGUF": "Controlled UNet Loader (GGUF)",
    "ControlledControlNetLoader": "Controlled ControlNet Loader",
    "ControlledVAELoader": "Controlled VAE Loader",
    "ControlledCLIPTextEncode": "Controlled CLIP Text Encode (Prompt)",
    "ControlledModelPatchLoader": "Controlled ModelPatch Loader",
    "SmartClipDeleter": "Smart CLIP Deleter (Auto-Detect)"
}
