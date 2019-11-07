package algorithms.control;

import algorithms.planning.MonteCarlo;
import algorithms.planning.PlanningAlgorithm;
import gridworld.GridWorld;
import policy.EGreedyPolicy;
import policy.Policy;
import utils.Vector2d;

/**
 * Created by dperez on 19/09/15.
 */
public class MonteCarloControl extends Control
{
    public double alpha;
    public MonteCarloControl(GridWorld gw, double epsilon, double gamma, double alpha)
    {
        super(gw, null, gamma);
        this.alpha = alpha;
        this.policy = new EGreedyPolicy(epsilon, false);
    }

    @Override
    public void execute() {
        //TODO 5: Implement one iteration of the Monte Carlo Control algorithm.
    }
}
