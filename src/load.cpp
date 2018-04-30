#include<fstream>
#include<vector>
#include<algorithm>
#include<iostream>
#include "load.h"
using namespace std;

bool mycomp(pair<int, int> a, pair<int , int> b){
    return (a.first<b.first || (a.first==b.first && a.second<b.second));
}

void loadFullGraph(const char * filename, uint32_t * oCSize, uint32_t * oRSize, uint32_t ** oFc, uint32_t ** oFr, uint32_t ** oBc, uint32_t ** oBr, uint32_t ** oPr){

  uint32_t  Edges = 0;
  uint32_t  Vertices = 0;
  char tmp[256];
  char tmp_c;
  uint32_t tmp_d;
  uint32_t  tmp_i, from, to;

  // open the file
  filebuf fb;
  fb.open(filename,ios::in);
  if (!fb.is_open() )
  {
     printf("Error Reading graph file\n");
     return;
  }
  istream is(&fb);

  // ignore the header of the file
  for(uint32_t i = 0; i<=6; i++)
    is.getline(tmp,256);

  //obtain the size of the graph (Edges, Vertices)
  is >> tmp_c >> tmp >> Vertices >> Edges;

    vector<pair<int, int> > edgeList;
    pair<int, int> p;


    for(unsigned int k=0;k<Edges;k++){
        is >> tmp_c >> p.first >> p.second >> tmp_i;
        edgeList.push_back(p);
    }

    sort(edgeList.begin(), edgeList.end(), mycomp);

  uint32_t  CSize = Edges;
  uint32_t  RSize = Vertices + 2;

  uint32_t* Fc = new uint32_t[CSize];
  uint32_t* Fr = new uint32_t[RSize];

  Fr[0] = 0;
  Fr[1] = 0;

  //obtain Fc, Fr
  uint32_t i = 1, j = 0;

  //cout<< "Reading the file" << endl;
  while(j < Edges){
    from = edgeList[j].first;
    to = edgeList[j].second;

     while(from > i)
     {
       Fr[ i + 1 ] = j;
       i++;
     }
     Fc[j] = to;
     j++;
  }

  //Fill up remaining indexes with M
  for(uint32_t k = i+1;k<RSize;k++)
    Fr[k] = j;

  //transposition
  uint32_t* Bc = new uint32_t[CSize];
  uint32_t* Br = new uint32_t[RSize];

  uint32_t * shift = new uint32_t [RSize];

  uint32_t target_vertex = 0, source_vertex = 0;

  for(unsigned int i = 0; i < RSize; i++)
  {
    Br[i] = 0;
    shift[i] = 0;
  }

  for(unsigned int i = 0; i < CSize; i++)
  {
    Br[Fc[i] + 1]++;
  }

  for(unsigned int i = 0; i < RSize - 1; i++)
  {
    Br[i+1] = Br[i] + Br[i+1];
  }

  for(unsigned int i = 0; i < CSize; i++)
  {
    while(i >= Fr[target_vertex + 1])
    {
       target_vertex++;
    }
    source_vertex = Fc[i];
    Bc[ Br[source_vertex] + shift[source_vertex] ] = target_vertex;
    shift[source_vertex]++;
  }
  delete [] shift;

  uint32_t * Pr = new uint32_t[RSize];

  filebuf fbm;
  fbm.open("//home//subbu//Desktop//PpP//src//MeTiSInput128.graph.part.10",ios::in);
  if (!fbm.is_open() )
  {
     printf("Error Reading metis graph file\n");
     return;
  }
  istream ism(&fbm);

  for(unsigned int k=0;k<Vertices;k++){
    ism >> tmp_d;
    Pr[k + 1] = tmp_d;
    //cout<<"Vertex: "<<k+1<<" "<<Pr[k + 1]<<endl;
  }


  *oCSize = Edges;
  *oRSize = Vertices;
  *oFc = Fc;
  *oFr = Fr;
  *oBc = Bc;
  *oBr = Br;
  *oPr = Pr;
}
