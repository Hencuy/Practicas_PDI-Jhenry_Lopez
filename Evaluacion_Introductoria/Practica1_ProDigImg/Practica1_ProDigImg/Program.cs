using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Practica1_ProDigImg
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Ingrese el numero que limite la serie de Fibonacci:");

            int num = Convert.ToInt32(Console.ReadLine());

            int a = 0, b = 1;

            for (int i = 0; i < num; i++)
            {
                Console.Write(a + " ");

                int aux = a;

                a = b;

                b = aux + b;
            }

            Console.ReadKey();
        }
    }
}
