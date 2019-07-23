#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

void print(const vector<int> &vi){
    cout<<"DEBUG: { ";
    for (auto x: vi) cout<<x<<", ";
    cout<<" }"<<endl;
}

void print(const vector<vector<int>::iterator> &vi){
    cout<<"DEBUG: { ";
    for (auto x: vi) cout<<*x<<", ";
    cout<<" }"<<endl;
}

static int pop(vector<int> &vi){
    int last = *(vi.end()-1);
    vi.pop_back();
    return last;
}

static int& top(vector<int> &vi){
    return *(vi.end()-1);
}

int main(){
    vector<int> vi={1,3,5,7};
    auto rst = find(vi.begin(), vi.end(),4); 
    cout<<"found: "<<*(rst)<<endl;
    vi.pop_back();
    reverse(vi.begin(), vi.end());
    print(vi);
    cout<<vi.size()<<endl;
}
