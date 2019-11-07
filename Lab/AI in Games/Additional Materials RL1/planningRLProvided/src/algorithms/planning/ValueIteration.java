package algorithms.planning;

import gridworld.GridWorld;
import policy.Policy;
import utils.Vector2d;

/**
 * Created by dperez on 22/08/15.
 */
public class ValueIteration extends PlanningAlgorithm {

    //Creates the value iteration class.
    public ValueIteration(GridWorld gw, Policy p, double gamma)
    {
        super(gw, p, gamma);

        //TODO 4: initialize the state-values v(s) for all s in the grid
    }

    @Override
    //Generates the next set of state-value values v(s). It should iterate through all states in the gridworld
    //and update the value of v(s) according to the value iteration update: v(s) = MAX{a in A} reward + gamma * v(s')
    public void execute()
    {
        //TODO 5b: Update the value of v(s) for all state-values, according to the algorithm's policy and update rule.
    }


    //Returns the action that, after being applied, takes the MDP to the state
    //with a highest value v(s)
    private Vector2d getActionWithMaxV(int row, int col)
    {
        //TODO 5a: Try all actions from the current state to reach a next state s'. Return the action
        //that achieves the highest v(s') value.
        return null;
    }



}
