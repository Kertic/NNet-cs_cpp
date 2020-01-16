using System;
using MathNet.Numerics.LinearAlgebra;

namespace NNet
{
    public class Layer
    {
        private Matrix<double> _weights;
        private Vector<double> _previousNeurons;
        private Vector<double> _manualNeurons;
        private Vector<double> _biases;
        public int NeuronCount { get; private set; }

        public Matrix<double> Weights
        {
            get { return _weights; }
            set
            {
                if (value.ColumnCount != _weights.ColumnCount || value.RowCount != _weights.RowCount)
                {
                    throw new Exception("Attempted to set weight matrix of size: {" + _weights.ColumnCount + "," +
                                        _weights.RowCount + "} to matrix of size: {" + value.ColumnCount + "," +
                                        value.RowCount + "}");
                }

                _weights = value;
            }
        }

        public Vector<double> Biases
        {
            get { return _biases; }
            set
            {
                if (_weights != null)
                    throw new Exception("Attempted to set neurons manually when they have a previous layer");
                if (_biases.Count != value.Count)
                    throw new Exception("Attempted to set neuron vector of size: " + _biases.Count +
                                        " with a constant vector of size: " + value.Count);
                _biases = value;
            }
        }

        public Vector<double> Neurons
        {
            get
            {
                Vector<double> weightedSums;
                if (_weights != null)
                {
                    weightedSums = _weights * (_previousNeurons + _biases);
                }
                else
                {
                    weightedSums = (_manualNeurons + _biases);
                }

                weightedSums.Map((d => d > 0 ? d : 0.0));
                return weightedSums;
            }
            set
            {
                if (_weights != null)
                    throw new Exception("Attempted to set neurons manually when they have a previous layer");
                if (NeuronCount != value.Count)
                    throw new Exception("Attempted to set neuron vector of size: " + NeuronCount +
                                        " with a constant vector of size: " + value.Count);
                _manualNeurons = value;
            }
        }

        public Layer(int neuronCount, Layer previousLayer = null)
        {
            NeuronCount = neuronCount;
            _biases = Vector<double>.Build.Dense(neuronCount);
            if (previousLayer == null)
            {
                _previousNeurons = null;
                _weights = null;
            }
            else
            {
                _previousNeurons = previousLayer.Neurons;
                _weights = Matrix<double>.Build.Dense(neuronCount, previousLayer.Neurons.Count);
            }
        }
    }
}