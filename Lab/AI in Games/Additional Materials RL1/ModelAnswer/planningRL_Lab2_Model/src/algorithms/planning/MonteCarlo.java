package algorithms.planning;

import gridworld.GridWorld;
import policy.Policy;

/**
 * This is First Visit, Constant-α MC
 * Created by dperez on 16/09/15.
 */
public class MonteCarlo extends PlanningAlgorithm
{
    public double alpha;

    public MonteCarlo(GridWorld gw, Policy p, double gamma, double alpha) {
        super(gw, p, gamma);

        this.alpha = alpha;
        for(int i = 0; i < value.length; ++i)
            for(int j = 0; j < value[0].length; ++j)
                this.value[i][j] = 0.0; //Values all 0.0
    }

    @Override
    //Generates the next set of state-value values v(s). It should iterate through all states in the gridworld
    //and update the value of v(s) according to the Monte Carlo update: v(s) = v(s) + alpha * (Return - v(s));
    public void execute() {

        //v(s) = v(s) + alpha * (Return - v(s));
        double[][] nextValue = new double[value.length][value[0].length];

        for(int i = 0; i < value.length; ++i)
            for(int j = 0; j < value[0].length; ++j)
            {
                double val = 0.0;
                if(gridWorld.isTerminal(i,j))
                    val = gridWorld.getReward(i,j);
                else
                {
                    double returnValue = this.getReturn(i, j);
                    double curValue = this.getValue(i,j);
                    val = curValue + alpha * (returnValue - curValue);
                }

                nextValue[i][j] = val;

                if(val > maxVal)
                    maxVal = val;
                if(val < minVal)
                    minVal = val;
            }

        for(int i = 0; i < value.length; ++i)
            System.arraycopy(nextValue[i], 0, value[i], 0, value.length);

    }
}
