using System.Collections.Generic;
using System.Linq;

namespace WatchAlyzer
{
    public class OcrResponse
    {
        public string Language { get; set; }
        public double TextAngle { get; set; }
        public string Orientation { get; set; }
        public List<Region> Regions { get; set; }

        private List<string> _texts = null;

        public List<string> Texts
        {
            get
            {
                if (_texts == null)
                {
                    _texts = new List<string>();

                    foreach (var r in Regions)
                    {
                        foreach (var l in r.Lines)
                        {
                            _texts.Add(string.Join(' ', l.Words.Select(w => w.Text)));
                        }
                    }
                }

                return _texts;
            }
        }
    }

    public class Region
    {
        public string BoundingBox { get; set; }
        public List<Line> Lines { get; set; }
    }

    public class Line
    {
        public string BoundingBox { get; set; }
        public List<Word> Words { get; set; }
    }

    public class Word
    {
        public string BoundingBox { get; set; }
        public string Text { get; set; }
    }
}
