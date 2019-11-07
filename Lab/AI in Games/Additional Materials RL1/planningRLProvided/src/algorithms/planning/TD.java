package algorithms.planning;

import gridworld.GridWorld;
import policy.Policy;
import utils.Vector2d;

/**
 * This is TD
 * Created by dperez on 16/09/15.
 */
public class TD extends PlanningAlgorithm
{
    public double alpha;

    public TD(GridWorld gw, Policy p, double gamma, double alpha) {
        super(gw, p, gamma);

        //TODO 8: initialize the state-values v(s) for all s in the grid
    }

    @Override
    //Generates the next set of state-value values v(s). It should iterate through all states in the gridworld
    //and update the value of v(s) according to the TD update: v(s) = v(s) + alpha * (Return + gamma * v(s') - v(s));
    public void execute() {
        //v(s) = v(s) + alpha * (Return + gamma * v(s') - v(s));
        //TODO 7: Update the value of v(s) for all state-values, according to the algorithm's policy and update rule.
    }

}
