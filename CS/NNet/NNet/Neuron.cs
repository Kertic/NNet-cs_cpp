using System;
using MathNet.Numerics.LinearAlgebra;

namespace NNet
{
    public class Neuron
    {
        private double _activation = 0.0f;
        public NeuronLink[] Influences;

        public double Activation
        {
            get
            {
                if (Influences == null || Influences.Length <= 0) return _activation;
                double totalVal = 0.0f;
                for (int i = 0; i < Influences.Length; i++)
                {
                    totalVal += Influences[i].GetWeightedSum();
                }

                var squish = (1.0f / (1 + Math.Pow(Math.E, -1.0f * totalVal)));
                _activation = squish;
                return totalVal;

            }
        }

        public Neuron(float inVal = 0.0f)
        {
            _activation = inVal;
        }
    }
}