import algorithms.control.Agent;
import algorithms.control.MonteCarloControl;
import algorithms.control.QLearning;
import algorithms.control.Sarsa;
import algorithms.planning.MonteCarlo;
import algorithms.planning.PlanningAlgorithm;
import algorithms.planning.TD;
import gridworld.GridWorld;
import gridworld.GridWorldCenter;
import policy.EGreedyPolicy;
import policy.GreedyPolicy;
import policy.Policy;
import policy.RandomPolicy;

import java.awt.*;

/**
 * Created by dperez on 22/08/15.
 */
public class Test
{
    public static void waitStep(int duration) {
        try
        {
            Thread.sleep(duration);
        }
        catch(InterruptedException e)
        {
            e.printStackTrace();
        }
    }

    public static void run()
    {
        Agent ag = new Agent();
        GridWorld gw = new GridWorldCenter(11); //new GridWorld(11);
        ag.setControl(gw);

        int numIterations = 1000;
        ag.train(numIterations);

        Point p = null;
        boolean justClicked = false;
        while(true)
        {
            Point clicked = ag.control.viewer.pointMouseClicked;
            if( (clicked != null) && (p != clicked)){
                justClicked = true;
                p = clicked;
            }

            if(justClicked){
                int row = (p.y / ag.control.viewer.cellSize);
                int col = p.x / ag.control.viewer.cellSize;
                ag.place(row, col);
                int i = 0;
                int limit = gw.size*2;
                while (!gw.isTerminal(ag.cRow, ag.cCol) && (i < limit)) {
                    ag.move();
                    i++;
                    //ag.control.draw();
                }
                justClicked = false;
            }

            ag.control.draw();
            waitStep(30);

        }
    }

    public static void main (String args[])
    {
        run();
    }
}
