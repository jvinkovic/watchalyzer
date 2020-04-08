using Microsoft.ML.Data;

namespace WatchAlyzer
{
    public class ImageLabelPredictions
    {
        [ColumnName(nameof(TensorFlowModelSettings.outputTensorName))]
        public string PredictedLabel;
    }
}
