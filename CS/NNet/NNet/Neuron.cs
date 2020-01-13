using MathNet.Numerics.LinearAlgebra;

namespace NNet
{
    public class Neuron
    {
        private float _value = 0.0f;
        public NeuronLink[] LinksFromPreviousLayer;

        public float Value
        {
            get
            {
                if (LinksFromPreviousLayer == null || LinksFromPreviousLayer.Length <= 0) return _value;
                float totalVal = 0.0f;
                for (int i = 0; i < LinksFromPreviousLayer.Length; i++)
                {
                    totalVal += LinksFromPreviousLayer[i].GetWeightedSum();
                }

                _value = totalVal;
                return totalVal;

            }
        }

        public Neuron(float inVal = 0.0f)
        {
            _value = inVal;
        }
    }
}