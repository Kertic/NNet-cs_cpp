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
            Layer test1 = new Layer(new Neuron[]
            {
                new Neuron(0.0f), 
                new Neuron(1.0f), 
                new Neuron(2.0f), 
            }, null);
            Layer test2 = new Layer(new Neuron[3], test1);

            Network testNet = new Network(new[]
            {
                test1, test2
            });
            testNet.RandomizeWeightsAndBiases();
            testNet.WriteWeightsAndBiases("output1.txt");
            testNet.ReadInWeightsAndBiases("output2.txt");

            Layer test3 = new Layer(new[] {new Neuron(1.0f), new Neuron(1.0f), new Neuron(1.0f)}, test1);

            testNet.Cost(test3);
        }
    }
}