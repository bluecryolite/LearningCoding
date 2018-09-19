package learning.lambda;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

public class Main {
    public static void main(String[] args) {
        LambdaAction action = new LambdaAction();
        Stream<Person> personStream = action.peek().filter(t -> t.getId() > 3);
        personStream.forEach(t -> {
            System.out.println(t.getJob());
        });

        action.partion().forEach((t, v) -> {
           System.out.println(t);
           System.out.println(v.size());
        });

        action.groupAndStat().forEach((t, v) -> {
            System.out.println(t);
            System.out.println(v.get().getName());
        });

        HashMap<String, Integer> result =
        action.helloMapReduce().get();
        result.forEach((t, v) -> {
            System.out.println(t + ":" + v.toString());
        });

        action.parallel();
    }
}
