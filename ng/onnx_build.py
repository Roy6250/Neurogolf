import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper

IR_VERSION = 10
OPSET_IMPORTS = [helper.make_opsetid("", 10)]

def make_tensor_value_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

def make_model_single_io(nodes, inits, input_shape=[1, 10, 30, 30], output_shape=[1, 10, 30, 30]):
    """Constructs a standard single-input single-output model for NeuroGolf."""
    x = make_tensor_value_info("input", input_shape)
    y = make_tensor_value_info("output", output_shape)
    
    graph = helper.make_graph(
        nodes=nodes,
        name="graph",
        inputs=[x],
        outputs=[y],
        initializer=inits
    )
    
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    # Validate
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model

def make_constant_node(name, output_name, value):
    """Helper to create a Constant node from a numpy array."""
    tensor = numpy_helper.from_array(value, name=name + "_val")
    node = helper.make_node("Constant", inputs=[], outputs=[output_name], name=name, value=tensor)
    return node
