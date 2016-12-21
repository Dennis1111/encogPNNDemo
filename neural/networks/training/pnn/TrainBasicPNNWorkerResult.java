package neural.networks.training.pnn;

public class TrainBasicPNNWorkerResult
{
  private double[] deriv;
  private double[] deriv2;
  private double error;
  
  public TrainBasicPNNWorkerResult(double error)
  {
    this.error=error;
    this.deriv=null;
    this.deriv2=null;
  }
  
  public void setDeriv(double[] deriv)
  {
    this.deriv=deriv;
  }

  public double[] getDeriv()
  {
    return deriv;
  }

  public void setDeriv2(double[] deriv2)
  {
    this.deriv2=deriv2;
  }

  public double[] getDeriv2()
  {
    return deriv2;
  }
    
  public double getError()
  {
    return error;
  }
}
