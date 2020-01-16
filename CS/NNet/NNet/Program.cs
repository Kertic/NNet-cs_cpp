using System;
using System.Diagnostics.Contracts;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace NNet
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            Layer testLayer = new Layer(3);
            testLayer.Neurons = Vector<double>.Build.Dense(new [] {1.0, 0.5, 0.0});
            Layer testLayer2 = new Layer(3,testLayer);
            testLayer2.Weights = Matrix<double>.Build.Dense(3, 3, 1.0);
            Console.WriteLine(testLayer2.Neurons);
            Layer[] testArr = new[]
            {
                new Layer(2),
                new Layer(2),
            };
            Network test = new Network(testArr);
        }
    }
}