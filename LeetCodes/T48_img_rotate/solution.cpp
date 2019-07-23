#include <iostream>
#include "../vector_utils.cpp"
typedef vector<vector<int>> mat;

void swap(int &a, int &b){
    int temp;
    temp = a;
    a = b;
    b = temp;
}

void transpose(mat &matrix){
    int n_row = matrix.size();
    int n_column = matrix[0].size();
    for (int i=0;i<n_row;++i){
        for(int j=i+1;j<n_column;++j){
            swap(matrix[i][j], matrix[j][i]);
        }
    }
}

void flip(mat &matrix){
    int n_row = matrix.size();
    int n_column = matrix[0].size();
    for(int i=0;i<n_row;++i){
        for (int j=0;j<n_column/2;++j){
            swap(matrix[i][j], matrix[i][n_column-1-j]);
        }
    }
}

int main()
{
    mat matrix;
    matrix.push_back(vector<int>({1,2,3,4}));
    matrix.push_back(vector<int>({5,6,7,8}));
    matrix.push_back(vector<int>({9,10,11,12}));
    matrix.push_back(vector<int>({13,14,15,16}));
    //matrix.push_back(vector<int>({1}));
    print(matrix);
    transpose(matrix);
    flip(matrix);
    cout<<"After rotation: "<<endl;
    print(matrix);
    return 0;
}

