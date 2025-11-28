# Compatibility shim to allow V3-style node definitions in a V1 ComfyUI environment
import inspect

class Schema:
    def __init__(self, node_id, display_name, category, inputs, outputs):
        self.node_id = node_id
        self.display_name = display_name
        self.category = category
        self.inputs = inputs
        self.outputs = outputs

class Input:
    def __init__(self, name, type_str, **kwargs):
        self.name = name
        self.type_str = type_str
        self.kwargs = kwargs

class Output:
    def __init__(self, type_str, name=None):
        self.type_str = type_str
        self.name = name

# Mock IO namespace
class io:
    Schema = Schema

    class Image:
        @staticmethod
        def Input(name, **kwargs): return Input(name, "IMAGE", **kwargs)
        @staticmethod
        def Output(name=None): return Output("IMAGE", name)

    class Int:
        @staticmethod
        def Input(name, **kwargs): return Input(name, "INT", **kwargs)
        @staticmethod
        def Output(name=None): return Output("INT", name)

    class Float:
        @staticmethod
        def Input(name, **kwargs): return Input(name, "FLOAT", **kwargs)
        @staticmethod
        def Output(name=None): return Output("FLOAT", name)

    class String:
        @staticmethod
        def Input(name, **kwargs): return Input(name, "STRING", **kwargs)
        @staticmethod
        def Output(name=None): return Output("STRING", name)

    class Enum:
        @staticmethod
        def Input(name, values, **kwargs):
            # values should be list
            return Input(name, values, **kwargs)

    class Any:
        # Not standard V3?
        pass

    class NodeOutput(tuple):
        pass

    class ComfyNode:
        @classmethod
        def define_schema(cls):
            raise NotImplementedError

        @classmethod
        def execute(cls, **kwargs):
            raise NotImplementedError

        # Magic to convert V3 schema to V1 INPUT_TYPES
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

            # We delay processing because define_schema might call other things?
            # Actually we can't call define_schema here if it relies on imports not ready.
            # But let's assume it's pure data.
            try:
                schema = cls.define_schema()
            except Exception:
                # Abstract class or error
                return

            # Generate V1 attributes
            cls.CATEGORY = schema.category
            # cls.TITLE = schema.display_name # Optional in V1, but good practice

            # INPUT_TYPES
            required = {}
            optional = {}

            for inp in schema.inputs:
                # Input definition: (Type, {options})
                type_def = inp.type_str
                options = inp.kwargs.copy()

                # Check for default to determine optionality?
                # V3 usually is explicit. Let's assume all required for now unless default is None?
                # Actually, ComfyUI V1 distinguishes required/optional.
                # If it has a default, is it required? Yes, just with a default.
                # Optional inputs are ones that can be disconnected.
                # For simplicity, we put everything in required unless explicitly "optional" flag (custom)
                # or if we decide logic based on default.

                required[inp.name] = (type_def, options)

            cls.INPUT_TYPES = classmethod(lambda s: {"required": required, "optional": optional})

            # RETURN_TYPES and RETURN_NAMES
            ret_types = []
            ret_names = []
            for out in schema.outputs:
                ret_types.append(out.type_str)
                if out.name:
                    ret_names.append(out.name)

            cls.RETURN_TYPES = tuple(ret_types)
            if ret_names:
                cls.RETURN_NAMES = tuple(ret_names)

            cls.FUNCTION = "execute_wrapper"

            # Wrapper to map kwargs
            def execute_wrapper(self, **kwargs):
                return cls.execute(**kwargs)

            setattr(cls, "execute_wrapper", execute_wrapper)

# Helper to check if we are in a real V3 env
try:
    from comfy_api.latest import io as real_io
    # If successful, we use the real one.
    # But wait, we want to write code that uses `io`.
    # So we should alias `io` to `real_io`.
    io = real_io
    USE_NATIVE_V3 = True
except ImportError:
    USE_NATIVE_V3 = False
