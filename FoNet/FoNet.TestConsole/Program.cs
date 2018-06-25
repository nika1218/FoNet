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
            var net = new Net(Function.Sigmoid, new Layer(2), new Layer(4), new Layer(1));
            TeachedNet(net);

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

        private static void TeachedNet(Net net)
        {
            for (int i = 0; i < teachSuit.Length; i++)
            {
            }
        }
    }
}
