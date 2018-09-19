package learning.lambda;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.*;

public class LambdaAction {
    private List<ImmutableTriple<Integer, String, String>> personDic;

    public LambdaAction() {
        personDic = new ArrayList<ImmutableTriple<Integer, String, String>>();
        personDic.add(new ImmutableTriple<Integer, String, String>(1, "Jesse", "Tester"));
        personDic.add(new ImmutableTriple<Integer, String, String>(3, "Tony", "Dever"));
        personDic.add(new ImmutableTriple<Integer, String, String>(2, "Maggie", "Tester"));
        personDic.add(new ImmutableTriple<Integer, String, String>(5, "Bary", "Dever"));
        personDic.add(new ImmutableTriple<Integer, String, String>(4, "Bob", "Dever"));
    }


    /**
     * 类型转换。通过循环，把一个集合转成另一个集合
     * 思路在创建新列表对象、循环、转换中切换，add那一句话容易写遗漏
     *
     * @return
     */
    public List<Person> convertOld() {
        List<Person> persons = new ArrayList<Person>();
        for(ImmutableTriple<Integer, String, String> t: personDic) {
            Person person = new Person();
            person.setId(t.getLeft());
            person.setName(t.getMiddle());
            person.setJob(t.getRight());
            persons.add(person);
        }

        return persons;
    }

    /**
     * 类型转换。通过Stream.map，把一个集合转换成为另一个集合
     * 思路只集中在转换上，不需要顾及其他。return那句容易写遗漏，但是会有编译错误，IDE也会立即指出错误
     *
     * @return
     */
    public List<Person> convert() {
        return convertInter().collect(Collectors.toList());
    }

    private Stream<Person> convertInter() {
        return personDic.stream().map(t -> {
            Person person = new Person();
            person.setId(t.getLeft());
            person.setName(t.getMiddle());
            person.setJob(t.getRight());
            return person;
        });
    }

    /**
     * 筛选
     *
     *
     * @return
     */
    public List<Person> filterOld() {
        List<Person> persons = new ArrayList<Person>();
        for(Person t: convertOld()) {
            if (t.getId() > 1) {
                persons.add(t);
            }
        }

        return persons;
    }

    /**
     * 筛选
     *
     * @return
     */
    public List<Person> filter() {
        return convertInter()
                .filter(t -> t.getId() > 1)
                .collect(Collectors.toList());
    }

    /**
     * 排序。
     * 多字段的写法：Comparator.comparing().thenComparing()
     *
     * @return
     */
    public List<Person> sort() {
        return convertInter().sorted(Comparator.comparing(Person::getId))
                .collect(Collectors.toList());
    }

    /**
     * 排序，倒序。
     *
     * @return
     */
    public List<Person> sortDesc() {
        return convertInter().sorted(Comparator.comparing(Person::getId).reversed())
                .collect(Collectors.toList());
    }

    /**
     * 分组
     * 多字段的写法：.collect(groupingBy(Class:field, groupingBy(Class:field)))
     *
     * @return
     */
    public Map<String, List<Person>> group() {
        return convertInter()
                .collect(groupingBy(Person::getJob));
    }

    /**
     * 分组并统计
     *
     * @return
     */
    public Map<String, Optional<Person>> groupAndStat() {
        return convertInter()
                .collect(groupingBy(Person::getJob, Collectors.maxBy(Comparator.comparing(Person::getId))));
    }

    /**
     * 分区
     * 多字段的写法：.collect(groupingBy(Class:field, groupingBy(Class:field)))
     *
     * @return
     */
    public Map<Boolean, List<Person>> partion() {
        return convertInter()
                .collect(partitioningBy(y -> ((Person)y).getId() > 2));
    }

    public Stream<Person> peek() {
        return convertInter().peek(t -> System.out.println(t.getName()));
    }

    public List<Person> flatMap() {
        //准备数据
        List<Person> sourcePersons = convertOld();
        List<Person> splitPersons1 = sourcePersons.stream().filter(t -> t.getId() < 3).collect(Collectors.toList());
        List<Person> splitPersons2 = sourcePersons.stream().filter(t -> t.getId() >= 3).collect(Collectors.toList());

        //扁平化集合的集合
        Stream<List<Person>> sourceLists = Stream.of(splitPersons1, splitPersons2);
        return sourceLists.flatMap(t -> t.stream()).collect(Collectors.toList());
    }

    /**
     * 统计多项值
     *
     * @return
     */
    public Optional<HashMap<String, Integer>> stat() {
        return convertInter()
                .map(t -> {
                    HashMap<String, Integer> result = new HashMap<String, Integer>();
                    result.put("min", t.getId());
                    result.put("max", t.getId());
                    result.put("count", 1);
                    return result;
                })
                .reduce((t, r) -> {
                    if (t.get("min") > r.get("min")) {
                        t.put("min", r.get("min"));
                    }
                    if (t.get("max") < r.get("max")) {
                        t.put("max", r.get("max"));
                    }
                    t.put("count", t.get("count") + 1);
                    return t;
                });
    }

    /**
     * 使用MapReduce做词频统计
     *
     * @return
     */
    public Optional<HashMap<String, Integer>> helloMapReduce() {
        //数据准备
        String sources = "This is a test file. This file is typed some persons: Tony Wu, Tony Wang, Tony Tang, Leon Liu, Leon Zhu, Tom Tang";
        Pattern pattern = Pattern.compile("[a-zA-Z]+");
        Matcher matcher = pattern.matcher(sources);
        Stream.Builder<String> builder = Stream.builder();
        while (matcher.find()) {
            builder.accept(matcher.group().toLowerCase());
        }

        //词频统计
        return builder.build().parallel().map(t -> {
            HashMap<String, Integer> result = new HashMap<String, Integer>();
            result.put(t, 1);
            return result;
        })
        .reduce((t, v) -> {
            //前面使用了并行，所以这里需要考虑做合并，即：t和v中都可能存在多条数据。
            for (String v_key : v.keySet()) {
                Integer n = t.get(v_key);
                if (n == null) {
                    n = 0;
                }
                t.put(v_key, v.get(v_key) + n);
            }

            return t;
        });
    }

    /**
     * 并行调用数据操作或者接口操作等其它方法
     *
     */
    public void parallel() {
        convertInter().parallel().peek(t -> t.setJob("new " + t.getJob()))
                .forEach(t -> System.out.println(t.getName() + " " + t.getJob()));
    }

    public void parallelByExecutor() {
       ExecutorService multiTaskExecutor = Executors.newFixedThreadPool(4);
       List<Future<ImmutablePair<Integer, String>>> newJobFutures = new ArrayList<Future<ImmutablePair<Integer, String>>>();
       for (Person person: convertOld()) {
           newJobFutures.add(multiTaskExecutor.submit(new NewJobCallable(person.getId(), person.getJob())));
       }

       for (Future<ImmutablePair<Integer, String>> item: newJobFutures) {
           try {
               ImmutablePair<Integer, String> response = item.get();
               //doSomething(response);
           } catch(Exception err) {
               err.printStackTrace();
           }
       }
    }

    private class NewJobCallable implements Callable<ImmutablePair<Integer, String>> {
        private final Integer id;
        private final String job;

        private NewJobCallable(Integer id, String job) {
            this.id = id;
            this.job = job;
        }

        @Override
        public ImmutablePair<Integer, String> call() {
            return new ImmutablePair<Integer, String>(id, "new " + job);
        }
    }
}
