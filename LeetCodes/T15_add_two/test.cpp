#include<iostream>
#include<vector>
using namespace std;
int main(){
    vector<int> vi={1,2,3,4,5,5};
    int beg=0;
    int end=vi.size()-1;
    while(beg!=end) {
        --end;
        cout<<end<<endl;
        if(vi[end]!=vi[end-1]) break;
    }
}
