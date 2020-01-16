using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

// ReSharper disable PossibleNullReferenceException

namespace NNet
{
    public class Network
    {
        private Layer[] _layers;

        public Network(Layer[] layers)
        {
            _layers = layers;
        }
    }
}