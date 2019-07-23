#include<vector>
#include<iostream>
using namespace std;

void print(const vector<int> &vi){
    cout<<"{ ";
    for (auto x: vi) cout<<x<<", ";
    cout<<" }"<<endl;
}

void print(const vector<vector<int>::iterator> &vi){
    cout<<"{ ";
    for (auto x: vi) cout<<*x<<", ";
    cout<<" }"<<endl;
}


vector<int> nextPermutation(vector<int>& nums) {
    vector<vector<int>::iterator> temp={nums.end()-1};
    auto last = nums.end()-2;
    auto begin = nums.cbegin();
    while(last >= begin){
        //cout<<"*last="<<*last<<", *temp[-1]="<<**(temp.end()-1)<<endl;
        if(*last >= **(temp.end()-1)){
            temp.push_back(last);
            //cout<<"push: "<<*last<<" size: "<<temp.size()<<endl;
            //cout<<"temp: ";
            //print(temp);
            if(last!=begin) --last;
            else break;
        } else {
            int store_value = *last;
            *last = *(temp[0]);
            *(temp[0]) = store_value;
            //cout<<"*last="<<*last<<", *temp[0]="<<*(temp[0])<<endl;
            break;
        }
    }
    vector<int> new_nums;
    for(int i=0;i<nums.size();++i){
        new_nums.push_back(*(temp[i]));
    }
    return new_nums;
    //nums = temp;
}

int main(){
    //vector<int> vi = {1,2,3,5,4};
    vector<int> vi = {1,3,2};
    print(vi);
    vi = nextPermutation(vi);
    print(vi);
}
