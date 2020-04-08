using System;
using System.IO;
using System.Security.Permissions;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Drawing;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision.Models;
using System.Threading.Tasks;
using System.Net.Http;
using System.Net.Http.Headers;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;

namespace WatchAlyzer
{
    internal class Program
    {
        private const string TensorFlowModelFilePath = "TFOCRModel.zip";

        private static MLContext _mlContext = new MLContext();
        private static ITransformer _mlModel;

        private static HttpClient httpClient = new HttpClient();

        private static string dir;
        private static string ext;

        private static List<FileSystemWatcher> watchers = new List<FileSystemWatcher>();

        private static void Main()
        {
            bool validDir = false;
            do
            {
                Console.Write("Directory to watch: ");
                dir = Console.ReadLine().Trim();

                if (Directory.Exists(dir))
                {
                    validDir = true;
                }
            } while (!validDir);

            int ans = 'y';
            Console.WriteLine("Watch subfolders? (Y/n)");
            ans = Console.Read();

            bool subfolders = ans == 'y';

            Console.Write("Files filters to watch (e.g.: \"*.jpg,*bmp*\"): ");
            ext = Console.ReadLine().Trim();

            Console.WriteLine("Starting model...");

            //_mlModel = SetupMlnetModel(TensorFlowModelFilePath);

            Console.WriteLine("Starting watch...");

            // make watcher
            StartWatcher(dir, ext.Split(','), subfolders);

            // Wait for the user to quit the program.
            Console.WriteLine("\nPress 'q' to quit\n\n");
            while (Console.Read() != 'q') ;
        }

        [PermissionSet(SecurityAction.Demand, Name = "FullTrust")]
        private static void StartWatcher(string dir, string[] extensions, bool includeSubDirs)
        {
            foreach (var ext in extensions)
            {
                var watcher = new FileSystemWatcher(dir, ext);

                // Watch for changes in LastAccess and LastWrite times
                watcher.NotifyFilter = NotifyFilters.LastWrite
                                    | NotifyFilters.LastAccess
                                    | NotifyFilters.CreationTime
                                    | NotifyFilters.FileName;

                watcher.IncludeSubdirectories = includeSubDirs;

                // Add event handlers.
                watcher.Created += OnChanged;
                watcher.Renamed += OnChanged;

                // Begin watching.
                watcher.EnableRaisingEvents = true;
            }

            // let user know what is watched
            Console.WriteLine($"\nWatching: {dir}\n \tincludeSubDirs: {includeSubDirs}\n  - Filters: {string.Join(", ", extensions)}\n");
        }

        private static void OnChanged(object source, FileSystemEventArgs e)
        {
            // get file and send it for processing
            Console.WriteLine(e.Name + " " + e.ChangeType);

            Task.Factory.StartNew(() => ProcessWithAzureOCR(e.FullPath), TaskCreationOptions.LongRunning | TaskCreationOptions.PreferFairness).GetAwaiter().GetResult();
        }

        private static async Task ProcessWithAzureOCR(string path)
        {
            /*
             AzureCognitiveServicesKey.txt:

                key=67kkiab7894esfsd4bb67b1bb4512k0                     --- this is fake example
                endpoint=https://SOMETH.api.cognitive.microsoft.com/    --- this is fake example
             */

            // Add your Computer Vision subscription key and endpoint
            var lines = File.ReadAllLines("AzureCognitiveServicesKey.txt");
            string key = lines[0].Split('=')[1];
            string endpoint = lines[1].Split('=')[1];

            httpClient.DefaultRequestHeaders.Add("Ocp-Apim-Subscription-Key", key);

            MemoryStream memoryStream = new MemoryStream();

            using (Bitmap img = Image.FromFile(path) as Bitmap)
            {
                Rectangle cropRect = new Rectangle(0, img.Height / 3 * 2, img.Width, img.Height / 3);
                using (Bitmap target = new Bitmap(cropRect.Width, cropRect.Height))
                {
                    using (Graphics g = Graphics.FromImage(target))
                    {
                        g.DrawImage(img, new Rectangle(0, 0, target.Width, target.Height),
                                         cropRect,
                                         GraphicsUnit.Pixel);
                    }

                    target.Save(memoryStream, System.Drawing.Imaging.ImageFormat.Png);
                    memoryStream.Position = 0;

                    //target.Save(path.Replace("jpg", "") + "-SMALL.png", System.Drawing.Imaging.ImageFormat.Png);
                }
            }

            string json = string.Empty;
            using (var content = new StreamContent(memoryStream))
            {
                content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                var response = await httpClient.PostAsync(endpoint + "vision/v2.0/ocr", content);
                json = await response.Content.ReadAsStringAsync();
            }

            //Console.WriteLine(JToken.Parse(json).ToString(Formatting.Indented));

            var ocrResult = JsonConvert.DeserializeObject<OcrResponse>(json);
            Console.WriteLine($"\npath: {path}");
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Write("Text:");
            foreach (var text in ocrResult.Texts)
            {
                if (text.ToLowerInvariant().Contains("code"))
                {
                    Console.ForegroundColor = ConsoleColor.DarkBlue;
                    Console.BackgroundColor = ConsoleColor.Green;
                }
                Console.Write($"\t {text}");
                Console.BackgroundColor = ConsoleColor.Black;
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine();
            }
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine();

            memoryStream.Dispose();
        }

        private static void ProcessImage(string path)
        {
            Bitmap bitmapImage = (Bitmap)Image.FromFile(path);

            //Set the specific image data into the ImageInputData type used in the DataView
            ImageInputData imageInputData = new ImageInputData { Image = bitmapImage };

            var predictor = _mlContext.Model.CreatePredictionEngine<ImageInputData, ImageLabelPredictions>(_mlModel);
            var prediction = predictor.Predict(imageInputData);

            Console.WriteLine($"Image: {Path.GetFileName(path)}\n\t - predicted as: {prediction.PredictedLabel}");
        }

        private static ITransformer SetupMlnetModel(string tensorFlowModelFilePath)
        {
            var pipeline1 = _mlContext.Transforms.ResizeImages(outputColumnName: TensorFlowModelSettings.inputTensorName,
                                                      imageWidth: ImageSettings.ImageWidth,
                                                      imageHeight: ImageSettings.ImageHeight,
                                                      inputColumnName: nameof(ImageInputData.Image));

            var pipeline2 = pipeline1.Append(_mlContext.Transforms.ExtractPixels(outputColumnName: TensorFlowModelSettings.inputTensorName,
                                                              interleavePixelColors: true,
                                                              offsetImage: 117))
                                    .Append(_mlContext.Transforms.ConvertToGrayscale(outputColumnName: TensorFlowModelSettings.inputTensorName));

            var pipeline3 = pipeline2.Append(_mlContext.Model.LoadTensorFlowModel(tensorFlowModelFilePath)
                                                       .ScoreTensorFlowModel(
                                                              outputColumnNames: new[] { TensorFlowModelSettings.outputTensorName },
                                                              inputColumnNames: new[] { TensorFlowModelSettings.inputTensorName },
                                                              addBatchDimensionInput: true));

            ITransformer mlModel = pipeline3.Fit(CreateEmptyDataView());

            SaveMLNetModel(mlModel, "MLNetModel.zip");

            return mlModel;
        }

        private static IDataView CreateEmptyDataView()
        {
            //Create empty DataView ot Images. We just need the schema to call fit()
            List<ImageInputData> list = new List<ImageInputData>();
            list.Add(new ImageInputData() { Image = new Bitmap(ImageSettings.ImageWidth, ImageSettings.ImageHeight) });

            var dv = _mlContext.Data.LoadFromEnumerable(list);
            return dv;
        }

        private static void SaveMLNetModel(ITransformer mlModel, string mlnetModelFilePath)
        {
            // Save/persist the model to a .ZIP file to be loaded by the PredictionEnginePool
            _mlContext.Model.Save(mlModel, null, mlnetModelFilePath);
        }
    }
}
