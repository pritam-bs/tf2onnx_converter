import tensorflow as tf
from keras.models import load_model
from keras.models import Model
import tf2onnx

onnx_model_name = 'model.onnx'

model: Model = load_model('glint360k_cosface_r100_fp16_0.1.h5')
new_batch_size = 1
new_input_shape = (new_batch_size,) + model.input_shape[1:]
spec = (tf.TensorSpec(new_input_shape, model.input.dtype, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(
    model=model, input_signature=spec, opset=13, output_path=onnx_model_name)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)
