package algorithms.planning;

import gridworld.GridWorld;
import policy.Policy;
import utils.Vector2d;

/**
 * Created by dperez on 22/08/15.
 */
public class ValueIteration extends PlanningAlgorithm {

    //Creates the value iteration class.
    public ValueIteration(GridWorld gw, Policy p)
    {
        super(gw, p, 0.0); //no need for gamma

        //Initialization
        for(int i = 0; i < value.length; ++i)
            for(int j = 0; j < value[0].length; ++j)
                //Initialization doesn't matter:
                //this.value[i][j] = rnd.nextDouble() - 0.5; //Values [-0.5 .. 0.5]
                this.value[i][j] = 0.0; //Values all 0.0
    }


    @Override
    //Generates the next set of state-value values v(s). It should iterate through all states in the gridworld
    //and update the value of v(s) according to the value iteration update: v(s) = SUM ( pi(a|s) * (reward + v(s'))
    public void execute()
    {
        double[][] nextValue = new double[value.length][value[0].length];

        for(int i = 0; i < value.length; ++i)
            for(int j = 0; j < value[0].length; ++j)
            {
                double val = 0.0;
                if(gridWorld.isTerminal(i,j))
                    val = gridWorld.getReward(i,j);
                else
                {
                    val = 0.0;
                    for(Vector2d v : gridWorld.actions)
                    {
                        double pProb = policy.prob(this.value, i, j, v, gridWorld.actions);
                        double thisActionVal = pProb * (gridWorld.getReward(i, j) + this.getValue(i, j, v));
                        val += thisActionVal;
                    }
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
