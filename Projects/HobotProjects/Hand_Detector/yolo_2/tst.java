public class HelloWorld {
    public static void main(String []args) {
        Test test = new Test();
        int i = test.calculate(5);
        System.out.println(i);
    }
    public int calculate(int a){
        static int b = 10;
        for (int i=0;i<a;i++){
            ++b;
        }
        return b;
    }
}
