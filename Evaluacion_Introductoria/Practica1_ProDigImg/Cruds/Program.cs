using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cruds
{
    class Estudiante
    {
        public int Id { get; set; }
        public string Nombre { get; set; }
        public string Apellido { get; set; }
        public int Edad { get; set; }


        public Estudiante(int id, string nombre, string apellido, int edad)
        {
            Id = id;
            Nombre = nombre;
            Apellido = apellido;
            Edad = edad;
        }
    }

    internal class Program
    {
        static Estudiante[] estudiantes = new Estudiante[50];
        static int cont = 0;
        static void Main(string[] args)
        {
            bool salir = true;

            while (salir)
            {
                Console.WriteLine("Seleccione una opción:");
                Console.WriteLine("==============================================");

                Console.WriteLine("1. Agregar");
                Console.WriteLine("2. Mostrar");
                Console.WriteLine("3. Actualizar");
                Console.WriteLine("4. Eliminar");
                Console.WriteLine("5. Salir");

                Console.WriteLine("==============================================");

                int opcion = Convert.ToInt32(Console.ReadLine());


                switch (opcion)
                {
                    case 1:
                        AgregarEstudiante();
                        break;
                    case 2:
                        MostrarEstudiantes();
                        break;
                    case 3:
                        //ActualizarEstudiante();
                        break;
                    case 4:
                        EliminarEstudiante();
                        break;
                    case 5:
                        salir = false;
                        break;
                    default:
                        Console.WriteLine("Seleccione una opcion valida");
                        break;
                }
            }
        }

        static void AgregarEstudiante()
        {
            Console.WriteLine("Nombre:");
            string nombre = Console.ReadLine();

            Console.WriteLine("Apellido:");
            string apellido = Console.ReadLine();

            Console.WriteLine("Edad:");

            int edad = Convert.ToInt32(Console.ReadLine());


            Estudiante estudiante = new Estudiante(cont + 1, nombre, apellido, edad);
            estudiantes[cont] = estudiante;
            cont++;

            Console.WriteLine("El estudiante se agrego");
        }

        static void MostrarEstudiantes()
        {
            Console.WriteLine("Listado de estudiantes:");
            foreach (var estudiante in estudiantes)
            {
                if (estudiante != null)
                {
                    Console.WriteLine("ID: " + estudiante.Id + " Nombre:" + estudiante.Nombre + " Apellido: " + estudiante.Apellido + " Edad: " + estudiante.Edad);
                }
            }
        }

        static void EliminarEstudiante()
        {
            Console.WriteLine("Ingrese el ID del estudiante que desea eliminar:");
            int id = Convert.ToInt32(Console.ReadLine());

            int indiceEliminar = -1;
            for (int i = 0; i < cont; i++)
            {
                if (estudiantes[i].Id == id)
                {
                    indiceEliminar = i;
                    break;
                }
            }

            if (indiceEliminar != -1)
            {
                for (int i = indiceEliminar; i < cont - 1; i++)
                {
                    estudiantes[i] = estudiantes[i + 1];
                }
                estudiantes[cont - 1] = null;
                cont--;

                Console.WriteLine("Estudiante eliminado");
            }
            else
            {
                Console.WriteLine("No se encontró ningún estudiante");
            }
        }
    }
}
