using Microsoft.ML.Transforms.Image;

using System.Drawing;

namespace WatchAlyzer
{
    public class ImageInputData
    {
        [ImageType(227, 227)]
        public Bitmap Image { get; set; }
    }
}
