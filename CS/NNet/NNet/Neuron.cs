using System;
using MathNet.Numerics.LinearAlgebra;

namespace NNet
{
    public class Neuron
    {
        private double _activation = 0.0f;
        private double _bias = 0.0f;
        private Vector<double> _influenceWeights;

        private NeuronLink[] _influences;
        public NeuronLink[] Influences
        {
            get { return _influences; }
            set
            {
                _influences = Influences;
                _influenceWeights = Vector<double>.Build.Dense(_influences.Length);
                for (int i = 0; i < Influences.Length; i++)
                {
                    _influenceWeights[i] = _influences[i].Weight;
                }
            }
        }

        public double Activation
        {
            get
            {
                if (Influences == null || Influences.Length <= 0) return _activation;
                double totalVal = _bias;
                for (int i = 0; i < Influences.Length; i++)
                {
                    totalVal += Influences[i].GetWeightedSum();
                }

                //double squish = (1.0f / (1 + Math.Pow(Math.E, -1.0f * totalVal))); //Sigmoid
                double squish = totalVal > 0.0 ? totalVal : 0.0; //reLU or Rectified Liner Activation Unit
                _activation = squish;
                return _activation;
            }
        }

        public Neuron(double inVal = 0.0, double bias = 0.0)
        {
            _activation = inVal;
            _bias = bias;
        }

        public Vector<double> GetInfluencesWeights()
        {
            Vector<double> returnVec = Vector<double>.Build.Dense(Influences.Length);
            for (int i = 0; i < returnVec.Count; i++)
            {
                returnVec[i] = Influences[i].Weight;
            }
            //Make a function that can set these as a vector as well
            return returnVec;
        }

        public void SetInfluencesWeights(Vector<double> newWeights)
        {
            _influenceWeights = newWeights;
            for (int i = 0; i < Influences.Length; i++)
            {
                _influences[i].Weight = _influenceWeights[i];
            }
        }
    }
}