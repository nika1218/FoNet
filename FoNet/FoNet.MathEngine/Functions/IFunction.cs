namespace FoNet.MathEngine.Functions
{
    public interface IFunction
    {
        float Apply(float val);

        float Derevative(float val);

        float Integral(float val);
    }
}
