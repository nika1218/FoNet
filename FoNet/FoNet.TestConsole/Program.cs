using System;
using FoNet.ANN;
using FoNet.MathEngine.Primitives;

namespace FoNet
{
    class Program
    {
        private static readonly int[][] teachSuit =
        {
            new []{0,0,0},
            new []{0,1,1},
            new []{1,0,1},
            new []{1,1,0}
        };

        static void Main(string[] args)
        {
            var net = new Net(Function.Sigmoid, new Layer(2), new Layer(5), new Layer(1));
            TeachNet(net);

            //net.Save(@"d:\temp\fonet.xml");

            Console.WriteLine("Data: {0,0} (0)");
            var data = net.Calculate(new float[] {0,0});
            Console.WriteLine("Result: {0}", data[0]);

            Console.WriteLine("Data: {0,1} (1)");
            data = net.Calculate(new float[] {0,1});
            Console.WriteLine("Result: {0}", data[0]);

            Console.WriteLine("Data: {1,0} (1)");
            data = net.Calculate(new float[] {1,0});
            Console.WriteLine("Result: {0}", data[0]);

            Console.WriteLine("Data: {1,1} (0)");
            data = net.Calculate(new float[] {1,1});
            Console.WriteLine("Result: {0}", data[0]);

            Console.ReadLine();
        }

        private static void TeachNet(Net net)
        {
            Console.WriteLine("Teaching started");
            for (int e = 0; e < 10; e++)
            {
                Console.WriteLine("\tEpoch {0}", e);

                for (int i = 0; i < teachSuit.Length; i++)
                {
                    var errors = net.CorrectErrors(0.9F, new float[] { teachSuit[i][0], teachSuit[i][1] }, new float[]{ teachSuit[i][2] });
                    Print($"Iteration {i} errors", errors, "\t\t");
                }
            }
            Console.WriteLine("Teaching finished");
        }

        private static void Print(string msg, float[] vector, string indent = null)
        {
            Console.WriteLine(indent + msg);
            foreach (var f in vector)
            {
                Console.Write("{1}{0:00.000} ", f, indent);
            }
            Console.WriteLine();
        }
    }
}
