# Compatibility shim to allow V3-style node definitions in a V1 ComfyUI environment
import inspect
import sys

# Attempt to import the real ComfyUI V3 API
try:
    from comfy_api.latest import io as real_io
    USE_NATIVE_V3 = True
    io = real_io
except ImportError:
    USE_NATIVE_V3 = False

    # Fallback Shim Implementation
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
                return Input(name, values, **kwargs)

        class Any:
             @staticmethod
             def Input(name, **kwargs): return Input(name, "*", **kwargs)

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

                try:
                    schema = cls.define_schema()
                except Exception:
                    return

                # Generate V1 attributes
                cls.CATEGORY = schema.category

                required = {}
                optional = {}

                for inp in schema.inputs:
                    type_def = inp.type_str
                    options = inp.kwargs.copy()

                    # Logic for required vs optional in V1
                    # In V1, if it has a default, it's often still in 'required'.
                    # 'optional' is usually for inputs that can be None/disconnected.
                    # We check 'optional' flag in kwargs which isn't standard V3 but we added it.
                    if options.pop("optional", False):
                        optional[inp.name] = (type_def, options)
                    else:
                        required[inp.name] = (type_def, options)

                cls.INPUT_TYPES = classmethod(lambda s: {"required": required, "optional": optional})

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

                def execute_wrapper(self, **kwargs):
                    return cls.execute(**kwargs)

                setattr(cls, "execute_wrapper", execute_wrapper)
