
import algorithms.planning.MonteCarlo;
import algorithms.planning.PlanningAlgorithm;
import algorithms.planning.TD;
import algorithms.planning.ValueIteration;
import gridworld.GridWorld;
import gridworld.GridWorldCenter;
import policy.Policy;
import policy.RandomPolicy;

import java.awt.*;

/**
 * Created by dperez on 22/08/15.
 */
public class Test
{

    public static void main (String args[])
    {
        GridWorld gw = new GridWorld(11);//new GridWorldCenter(11); //new GridWorld(11);

        double epsilon = 0.1;
        double gamma = 0.9;
        double alpha = 0.1;

        Policy p = new RandomPolicy();

        PlanningAlgorithm pAlg = new ValueIteration(gw, p, gamma);
        //PlanningAlgorithm pAlg = new MonteCarlo(gw, p, gamma, alpha);
        //PlanningAlgorithm pAlg = new TD(gw, p, gamma, alpha);

        long timestamp = System.currentTimeMillis();
        int k = 0;
        for( k = 0; k<= 100000; ++k) {
            //System.out.println(k);
            pAlg.execute();
            pAlg.draw();
        }

        long now = System.currentTimeMillis();
        System.out.println("Done, in " + (now-timestamp) + " ms.");

        for(int i =0; i < gw.size; i++) {
            for (int j = 0; j < gw.size; j++)
                System.out.format("%.10f ", pAlg.getValue(i, j));
            System.out.println();
        }

        //System.out.println("k = " + k);
        while(true) {
            pAlg.draw();
            waitStep(30);
        }

    }


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
}
