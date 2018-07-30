using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace FoNet.ANN
{
    internal class SynopsisMap
    {
        /// <summary>
        /// Map of W synapsis coefficients.
        /// </summary>
        /// <remarks>float[x,y] when x - inputDimension (previous layer output), y - outputDimension</remarks>
        private readonly IDictionary<ushort, float[,]> _wMap = new Dictionary<ushort, float[,]>();

        public SynopsisMap(params ushort[] layerDimensions)
        {
            if (layerDimensions == null) throw new ArgumentNullException(nameof(layerDimensions));
            if (layerDimensions.Length < 2) throw new ArgumentOutOfRangeException(nameof(layerDimensions), "Count of layers should be not less than two.");

            _wMap.Clear();
            // input for first layer map is its output
            _wMap[0] = new float[layerDimensions[0],layerDimensions[0]];

            for (ushort i = 1; i < layerDimensions.Length; i++)
            {
                var prevLayerDimension = layerDimensions[i - 1];
                var currentLayerDimension = layerDimensions[i];

                _wMap[i] = new float[prevLayerDimension, currentLayerDimension];
            }
        }

        public void Fill()
        {
            Parallel.For(0, _wMap.Count, i =>
            {
                var rnd = new Random();
                var layerMap = _wMap[(ushort)i];
                for (int x = 0; x < layerMap.GetLength(0); x++)
                {
                    for (int y = 0; y < layerMap.GetLength(1); y++)
                    {
                        // For the first layer all W=1
                        layerMap[x, y] = i == 0 ? 1 : (float)rnd.NextDouble() * 6 - 3; // [-3;3]
                    }
                }
            });
        }

        public void Fill(ushort layerId, float[,] newLayerMap)
        {
            ValidateLayerId(layerId);

            var layerMap = _wMap[layerId];
            ValidateWMapDimensions(layerMap, newLayerMap);

            _wMap[layerId] = newLayerMap;
        }

        public float[,] GetWMap(ushort layerId)
        {
            ValidateLayerId(layerId);

            return _wMap[layerId];
        }

        public IDictionary<ushort, float[,]> GetWMaps()
        {
            return _wMap;
        }

        private void ValidateLayerId(ushort layerId)
        {
            if (!_wMap.ContainsKey(layerId)) throw new ArgumentOutOfRangeException(nameof(layerId), $"There is no layer {layerId} in the map.");
        }

        private void ValidateWMapDimensions(float[,] layerMap, float[,] newLayerMap)
        {
            if (layerMap.GetLength(0) != newLayerMap.GetLength(0) || layerMap.GetLength(1) != newLayerMap.GetLength(1)) throw new ArgumentOutOfRangeException(nameof(layerMap), $"newLayerMap dimension [{newLayerMap.GetLength(0)},{newLayerMap.GetLength(1)}] doesn't corrspond to existing one [{layerMap.GetLength(0)},{layerMap.GetLength(1)}]");
        }
    }
}
