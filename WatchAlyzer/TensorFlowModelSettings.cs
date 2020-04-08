namespace WatchAlyzer
{
    // For checking tensor names, you can open the TF model .pb file with tools like Netron: https://github.com/lutzroeder/netron
    public struct TensorFlowModelSettings
    {
        // input tensor name
        public const string inputTensorName = "input_data";

        // output tensor name
        public const string outputTensorName = "y_pred";
    }
}
