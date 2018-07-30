using FoNet.MathEngine.Functions;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace FoNet.MathEngine
{
    public class CpuMathEngine : IMathEngine
    {
        public float[] Multiply(float[] vector, IDictionary<ushort, float[,]> matrices, IFunction activationFunction)
        {
            activationFunction = activationFunction ?? new Linear();

            var currentVector = vector;

            foreach (var currentMatrix in matrices)
            {
                currentVector = Multiply(currentVector, currentMatrix.Value, activationFunction);
            }

            return currentVector;
        }

        public float[] Multiply(float[] vector, float[,] matrix, IFunction activationFunction)
        {
            activationFunction = activationFunction ?? new Linear();

            var xLength = matrix.GetLength(0);
            var yLength = matrix.GetLength(1);

            if (vector.Length != xLength) throw new ArgumentOutOfRangeException(nameof(vector), $"Vector length {vector.Length} doesn't correspond to matrix size [{xLength},{yLength}]");

            var result = new float[yLength];

            Parallel.For(0, yLength, y =>
            {
                for (int x = 0; x < xLength; x++)
                {
                    result[y] += vector[x] * matrix[x, y];
                }

                result[y] = activationFunction.Apply(result[y]);
            });

            return result;
        }

        public float[] Errors(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length) throw new ArgumentOutOfRangeException(nameof(vector1), $"Vector 1 length {vector1.Length} doesn't correspond to vector 2 length {vector2.Length}");

            var result = new float[vector1.Length];

            Parallel.For(0, vector1.Length, i =>
            {
                result[i] += (float)Math.Pow(vector1[i] - vector2[i], 2);
            });

            return result;
        }

        #region ApplyFunction

        public float[] ApplyFunction(float[] vector, IFunction activationFunction)
        {
            activationFunction = activationFunction ?? new Linear();

            var result = new float[vector.Length];
            Parallel.For(0, vector.Length, i => { result[i] = activationFunction.Apply(vector[i]); });
            return result;
        }

        public float[] ApplyDerivativeFunction(float[] vector, IFunction activationFunction)
        {
            activationFunction = activationFunction ?? new Linear();

            var result = new float[vector.Length];
            Parallel.For(0, vector.Length, i => { result[i] = activationFunction.Derevative(vector[i]); });
            return result;
        }

        public float[] ApplyIntegralFunction(float[] vector, IFunction activationFunction)
        {
            activationFunction = activationFunction ?? new Linear();

            var result = new float[vector.Length];
            Parallel.For(0, vector.Length, i => { result[i] = activationFunction.Integral(vector[i]); });
            return result;
        }

        #endregion

        public float[] CorrectWeightsIteration(float epsilon, float[] vector, IDictionary<ushort, float[,]> matrices, float[] ideal,
            IFunction activationFunction)
        {
            float[][] inputs = new float[matrices.Count][];
            var linearFunction = new Linear();

            // forward pass
            var currentVector = vector;
            for (ushort i = 0; i < matrices.Count; i++)
            {
                inputs[i] = Multiply(currentVector, matrices[i], linearFunction);
                currentVector = ApplyFunction(inputs[i], activationFunction);
            }

            var errors = Errors(ideal, ApplyFunction(inputs[matrices.Count - 1], activationFunction));

            var sigmaPrev = SigmaOut(ideal, inputs[matrices.Count - 1], activationFunction);

            for (ushort i = (ushort)(matrices.Count - 1); i > 0; i--)
            {
                var wMap = matrices[i];
                var xLength = wMap.GetLength(0);
                var input = inputs[i];
                Parallel.For(0, sigmaPrev.Length, y =>
                {
                    for (var x = 0; x < xLength; x++)
                    {
                        var grad = activationFunction.Apply(input[y]) * sigmaPrev[y];
                        wMap[x, y] += grad * epsilon;
                    }
                });

                sigmaPrev = SigmaHidden(inputs[i - 1], matrices[i], sigmaPrev, activationFunction);
            }

            return errors;
        }

        private float[] SigmaOut(float[] ideal, float[] input, IFunction activationFunction)
        {
            activationFunction = activationFunction ?? new Linear();

            float[] sigmaOut = new float[input.Length];

            Parallel.For(0, sigmaOut.Length, i => { sigmaOut[i] = (ideal[i] - activationFunction.Apply(input[i])) * activationFunction.Derevative(input[i]); });

            return sigmaOut;
        }

        private float[] SigmaHidden(float[] input, float[,] wMapOut, float[] sigmaPrev, IFunction activationFunction)
        {
            activationFunction = activationFunction ?? new Linear();

            var xLength = wMapOut.GetLength(0);
            var yLength = wMapOut.GetLength(1);

            if (xLength != input.Length) throw new ArgumentOutOfRangeException(nameof(wMapOut));
            if (yLength != sigmaPrev.Length) throw new ArgumentOutOfRangeException(nameof(wMapOut));

            float[] sigmaHidden = new float[xLength];

            Parallel.For(0, xLength, x =>
            {
                var sigmaSum = 0F;
                for (var y = 0; y < yLength; y++)
                {
                    sigmaSum += wMapOut[x, y] * sigmaPrev[y];
                }
                sigmaHidden[x] = sigmaSum * activationFunction.Derevative(input[x]);
            });

            return sigmaHidden;
        }
    }
}
