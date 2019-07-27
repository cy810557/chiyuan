#include<iostream>
#include<vector>
using namespace std;

int binary_search(const vector<int> &vi, int e){
    int lo=0, hi=vi.size();
    while(hi-lo>1){
        int mid = (lo+hi)>>1;
        if(e<vi[mid]) hi=mid;
        else lo=mid;
    }
    return vi[lo]==e ? lo : -1;
}

int main(){
    vector<int> vi={1,2,3,4,5,6,7,8,9};
    int rst = binary_search(vi, 3);
    cout<<"result: "<<(rst==-1 ? -1 : vi[rst])<<endl;

}
