package algorithms.planning;

import gridworld.GridWorld;
import policy.Policy;
import utils.Vector2d;

/**
 * Created by dperez on 22/08/15.
 */
public class DynamicProgramming extends PlanningAlgorithm {

    //Creates the value iteration class.
    public DynamicProgramming(GridWorld gw, Policy p, double gamma)
    {
        super(gw, p, gamma); //no need for gamma

        //Initialization
        for(int i = 0; i < value.length; ++i)
            for(int j = 0; j < value[0].length; ++j)
                //Initialization doesn't matter:
                //this.value[i][j] = rnd.nextDouble() - 0.5; //Values [-0.5 .. 0.5]
                this.value[i][j] = 0.0; //Values all 0.0
    }


    @Override
    //Generates the next set of state-value values v(s). It should iterate through all states in the gridworld
    //and update the value of v(s) according to the value iteration update: v(s) = SUM p(a|s) (reward + gamma * V(s'))
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
                    for(Vector2d act : gridWorld.actions)
                    {
                        double prob = policy.prob(value, i, j, act, gridWorld.actions);
                        double nextReward = gridWorld.getReward(i, j);
                        val += prob * (nextReward + gamma * this.getValue(i, j, act));
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
