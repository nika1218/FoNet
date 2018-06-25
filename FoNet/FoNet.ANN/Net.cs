using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml.Serialization;
using FoNet.MathEngine;
using FoNet.MathEngine.Primitives;

namespace FoNet.ANN
{
    public class Net
    {

        private readonly IMathEngine _mathEngine = new CpuMathEngine();
        private readonly SynopsisMap _synopsisMap;

        public Net(ActivationFunction activationFunction, params Layer[] layers)
        {
            ActivationFunction = activationFunction;
            Layers = layers;
            _synopsisMap = new SynopsisMap(Layers.Select(l => l.NeuronsCount).ToArray());
            _synopsisMap.Fill();
        }

        public Net(string path)
        {
            // TODO loading map from file
        }

        public IList<Layer> Layers { get; set; }
        public ActivationFunction ActivationFunction { get; }

        public float[] Calculate(float[] input)
        {
            var result = _mathEngine.Multiply(input, _synopsisMap.GetWMaps(), ActivationFunction);
            result = _mathEngine.ApplyIntegralFunction(result, ActivationFunction);
            return result;
        }

        public void CorrectErrors(float[] deltas)
        {

        }

        public void Save(string path)
        {
            // TODO
            using (var file = File.CreateText(path))
            {
                var serializer = new XmlSerializer(typeof(SynopsisMap));
                serializer.Serialize(file, _synopsisMap);
            }
        }
    }
}
