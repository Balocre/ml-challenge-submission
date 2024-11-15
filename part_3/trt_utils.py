import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from onnx.external_data_helper import convert_model_to_external_data

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def cleanup_graph_for_trt_export(onnx_path):
    """A function to clean the graph of the ONNX model

    It is necessary to clean the graph of the DINOv2 model before export, because
    there are some artifact remaining from the pretraining that prevent us from
    converting to tensorRT engine.

    :param onnx_path: Path of the ONNX model file to cleanup.

    """

    graph = gs.import_onnx(onnx.load(onnx_path))
    graph.cleanup(
        remove_unused_node_outputs=True, remove_unused_graph_inputs=True
    ).toposort()

    model = gs.export_onnx(graph)
    convert_model_to_external_data(model, location="model.onnx.data")

    onnx.save(model, onnx_path)


def build_engine(onnx_path, trt_path="engine.trt"):
    """A function to build a TensorRt engine

    :param onnx_path: Path to the ONNX file defining the model
    :param trt_path: Name of the file for conversion
    """

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # set workspace memory to 8GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (2**30))

    # parse the ONNX file for errors
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError

        print("ONNX file parsed succesfully")

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("There was an error during serialization")

    with open(trt_path, "wb") as f:
        print("Serialzing engine to file")
        f.write(engine_bytes)
