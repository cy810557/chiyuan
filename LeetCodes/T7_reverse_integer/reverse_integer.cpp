#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

pair<int, int> divmod(int t1, int t2){
    return std::make_pair(t1/t2, t1%t2);
}

void divmod(int t1, int t2, int &quot, int &remi){
    quot = t1/t2;
    remi = t1%t2;
}

int reverse(int x){
    int flag_neg = 0;
    if(x<0) flag_neg=1;
}

int main(){
    int dig = 1234;
    cout<<dig/1000<<endl;
    cout<<dig%1000<<endl;
    vector<int> vint = {1,2,3,4,5,6};
    cout<<"======= func test:  std::minmax_element ======"<<endl;
    auto it = minmax_element(vint.begin(), vint.end());
    cout<<*it.first<<" "<<*it.second<<endl;
    cout<<"======= func test:  divmod(return pair) ======"<<endl;
    auto rst = divmod(dig, 1000);
    cout<<rst.first<<" "<<rst.second<<endl;
    cout<<"======= func test:  divmod(reference mode) ======"<<endl;
    int r1, r2;
    divmod(dig, 1000, r1, r2);
    cout<<"quot: "<<r1<<", remi: "<<r2<<endl;

}
