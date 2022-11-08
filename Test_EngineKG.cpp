// Test_EngineKG.cpp
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<sstream>
using namespace std;

bool debug=false;

string used = "1";

bool L1_flag=1;

map<pair<string,int>,double>  path_confidence;

vector<int> rel_type;

string version;
string trainortest = "test";
string res_path = "/dc_loop";
string data_dir = "fb15k";

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;
map<pair<int, int>, pair<int, double>> rule2rel;

vector<vector<pair<int,int> > > e1_e3;

int relation_num,entity_num;
int n= 100;
int Hits_N = 10;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%9==4)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}

class Test{
    vector<vector<double> > relation_vec,entity_vec;


    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;

	map<pair<int,int>,vector<pair<vector<int>,double> > >fb_path;
	
    map<pair<int,int>, map<int,int> > ok;
    double res;
public:
    void add(int x,int y,int z, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }

    void add(int x,int y,int z, vector<pair<vector<int>,double> > path_list)
    {
		if (z!=-1)
		{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        	ok[make_pair(x,z)][y]=1;
		}
		if (path_list.size()>0)
		fb_path[make_pair(x,y)] = path_list;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            sum+=-fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
        for (int ii=0; ii<n; ii++)
            sum+=-sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            sum+=-fabs(entity_vec[e1][ii]-entity_vec[e2][ii]-relation_vec[rel+relation_num][ii]);
        else
        for (int ii=0; ii<n; ii++)
            sum+=-sqr(entity_vec[e1][ii]-entity_vec[e2][ii]-relation_vec[rel+relation_num][ii]);
		int h = e1;
		int l = e2;
		if (used=="1")
		{
			vector<pair<vector<int>,double> > path_list = fb_path[make_pair(h,l)];
			int weightp = 30;
			if (path_list.size()>0)
			{
				for (int path_id = 0; path_id<path_list.size(); path_id++)
				{
					vector<int> rel_path = path_list[path_id].first;

					double pr_path = 0;
					double pr = path_list[path_id].second;
					int rel_integ;
					double confi_integ = 0;
					double confi_path = 1;
					string  s;
				    ostringstream oss;
					for (int ii=0; ii<rel_path.size(); ii++)
					{
						oss<<rel_path[ii]<<" ";
					}
				    s=oss.str();//
					if (path_confidence.count(make_pair(s,rel))>0)
						pr_path = path_confidence[make_pair(s,rel)];
					//pr_path = 1;
					// compose path via len-2 rules
					if (rel_path.size() > 1){
                                            for (int i = 0; i < rel_path.size(); i++){
                                                if (rule2rel.count(make_pair(rel_path[i], rel_path[i+1])) > 0){                                                    
                                                    rel_integ = rule2rel[make_pair(rel_path[i], rel_path[i+1])].first;
                                                    confi_integ = rule2rel[make_pair(rel_path[i], rel_path[i+1])].second;
                                                    confi_path = confi_path * confi_integ;
                                                    rel_path[i] = rel_integ;
                                                    for (int j = (i+1); j < (rel_path.size() - 1); j++){
                                                        rel_path[j] = rel_path[j+1];
                                                    }
                                                    rel_path.pop_back();
                                                }
                                            }
                                        }

					sum+=calc_path(rel,rel_path)*pr*pr_path*confi_path*weightp;
				}
			}
			path_list = fb_path[make_pair(l,h)];
			if (path_list.size()>0)
			{
				for (int path_id = 0; path_id<path_list.size(); path_id++)
				{
					vector<int> rel_path = path_list[path_id].first;
					double pr = path_list[path_id].second;
					double pr_path = 0;
					int rel_integ;
                                        double confi_integ = 0;
                                        double confi_path = 1;
					string  s;
				    ostringstream oss;//创建一个流
					for (int ii=0; ii<rel_path.size(); ii++)
					{
						oss<<rel_path[ii]<<" ";
					}
				    s=oss.str();//
					if (path_confidence.count(make_pair(s,rel+relation_num))>0)
						pr_path = path_confidence[make_pair(s,rel+relation_num)];
					//pr_path = 1;
					// compose path via len-2 rules
                                        if (rel_path.size() > 1){
                                            for (int i = 0; i < rel_path.size(); i++){
                                                if (rule2rel.count(make_pair(rel_path[i], rel_path[i+1])) > 0){
                                                    rel_integ = rule2rel[make_pair(rel_path[i], rel_path[i+1])].first;
                                                    confi_integ = rule2rel[make_pair(rel_path[i], rel_path[i+1])].second;
                                                    confi_path = confi_path * confi_integ;
                                                    rel_path[i] = rel_integ;
                                                    for (int j = (i+1); j < (rel_path.size() - 1); j++){
                                                        rel_path[j] = rel_path[j+1];
                                                    }
                                                    rel_path.pop_back();
                                                }
                                            }
                                        }

					sum+=calc_path(rel+relation_num,rel_path)*pr*pr_path*confi_path*weightp;
				}
			}
		}
        return sum;
    }
    double calc_path(int r1,vector<int> rel_path)
    {
        double sum=0;
        for (int ii=0; ii<n; ii++)
		{
			double tmp = relation_vec[r1][ii];
			for (int j=0; j<rel_path.size(); j++)
				tmp-=relation_vec[rel_path[j]][ii];
	        if (L1_flag)
				sum+=-fabs(tmp);
			else
				sum+=-sqr(tmp);
		}
        return (20+sum);
    }
    void run()
    {
        FILE* f1 = fopen(("./res/" + res_path + "/relation2vec_3.txt").c_str(),"r");
        FILE* f3 = fopen(("./res/" + res_path + "/entity2vec_3.txt").c_str(),"r");
        cout<<relation_num<<' '<<entity_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb*2);
        for (int i=0; i<relation_num_fb*2;i++)
        {
            relation_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
            if (vec_len(entity_vec[i])-1>1e-3)
            	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
        }
        fclose(f1);
        fclose(f3);
		
		
		
		double lsum=0 ,lsum_filter= 0;
		double rsum = 0,rsum_filter=0;
		double mid_sum = 0,mid_sum_filter=0;
		double lp_n=0,lp_n_filter = 0;
		double rp_n=0,rp_n_filter = 0;
		double mid_p_n=0,mid_p_n_filter = 0;
		map<int,double> lsum_r,lsum_filter_r;
		map<int,double> rsum_r,rsum_filter_r;
		map<int,double> mid_sum_r,mid_sum_filter_r;
		map<int,double> lp_n_r,lp_n_filter_r;
		map<int,double> rp_n_r,rp_n_filter_r;
		map<int,double> mid_p_n_r,mid_p_n_filter_r;
		map<int,int> rel_num;
		
		
		double l_one2one=0,r_one2one=0,one2one_num=0;
       		double l_n2one=0,r_n2one=0,n2one_num=0;
 	        double l_one2n=0,r_one2n=0,one2n_num=0;
 	        double l_n2n=0,r_n2n=0,n2n_num=0;
		
		double mrr_lsum_filter=0, mrr_rsum_filter=0, mrr_midsum_filter=0;
		double mrr_lsum=0, mrr_rsum=0, mrr_midsum=0;

		int hit_n = Hits_N;
		map<pair<int,int>,int> e1_e2;
		for (int testid = 0; testid<fb_l.size()/2; testid+=1)
		{
			int h = fb_h[testid*2];
			int l = fb_l[testid*2];
			int rel = fb_r[testid*2];
			rel_num[rel]+=1;
			vector<pair<int,double> > a;
			
            if (rel_type[rel]==0)
                one2one_num+=1;
            else
            if (rel_type[rel]==1)
                n2one_num+=1;
            else
            if (rel_type[rel]==2)
                one2n_num+=1;
            else
                n2n_num+=1;
			

			int filter = 0;
			for (int i = 0; i < entity_num; i++)
			{
				double sum = calc_sum(i, l, rel);
				a.push_back(make_pair(i, sum));
			}
			sort(a.begin(), a.end(), cmp);
			int rank_l = 0;
			for (int i=a.size()-1; i>=0; i--)
			{
			    if (ok[make_pair(a[i].first,rel)].count(l)==0)
			    	filter+=1;
				if (a[i].first ==h)
				{
					lsum+=a.size()-i;
					rank_l = a.size()-i;
					lsum_filter+=filter+1;
					lsum_r[rel]+=a.size()-i;
					lsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_n)
					{
						lp_n+=1;
						lp_n_r[rel]+=1;
					}
					if (filter<hit_n)
					{
						lp_n_filter+=1;
						lp_n_filter_r[rel]+=1;
						if (rel_type[rel]==0)
                            l_one2one+=1;
                        else
                        if (rel_type[rel]==1)
                            l_n2one+=1;
                        else
                        if (rel_type[rel]==2)
                            l_one2n+=1;
                        else
                            l_n2n+=1;
						
					}
					break;
				}
			}
			mrr_lsum_filter += 1/(double)(filter+1);
			mrr_lsum += 1/(double)(rank_l);
			a.clear();

			for (int i = 0; i < entity_num; i++)
			{
				double sum = calc_sum(h, i, rel);
				a.push_back(make_pair(i ,sum));
			}
			sort(a.begin(),a.end(),cmp);
			filter=0;
			int rank_r = 0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(h,rel)].count(a[i].first)==0)
			    	filter+=1;
				if (a[i].first==l)
				{
					rsum+=a.size()-i;
					rank_r = a.size()-i;
					rsum_filter+=filter+1;
					rsum_r[rel]+=a.size()-i;
					rsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_n)
					{
						rp_n+=1;
						rp_n_r[rel]+=1;
					}
					if (filter<hit_n)
					{
						rp_n_filter+=1;
						rp_n_filter_r[rel]+=1;
						if (rel_type[rel]==0)
							r_one2one+=1;
						else
						if (rel_type[rel]==1)
							r_n2one+=1;
						else
						if (rel_type[rel]==2)
							r_one2n+=1;
						else
							r_n2n+=1;
						
					}
					break;
				}
			}
			mrr_rsum_filter += 1/(double)(filter+1);
			mrr_rsum += 1/(double)(rank_r);
			a.clear();
			for (int i=0; i<relation_num; i++)
			{
				double sum = 0;
				sum+=calc_sum(h,l,i);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			filter=0;
			int rank_mid = 0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(h,a[i].first)].count(l)==0)
			    	filter+=1;
				if (a[i].first==rel)
				{
					mid_sum+=a.size()-i;
					rank_mid = a.size()-i;
					mid_sum_filter+=filter+1;
					mid_sum_r[rel]+=a.size()-i;
					mid_sum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_n)
					{
						mid_p_n+=1;
						mid_p_n_r[rel]+=1;
					}
					if (filter<hit_n)
					{
						mid_p_n_filter+=1;
						mid_p_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			mrr_midsum_filter += 1/(double)(1+filter);
			mrr_midsum += 1/(double)(rank_mid);
			if (testid%100==0)
			{
				cout<<testid<<":"<<"\t"<<lsum/(testid+1)<<' '<<lp_n/(testid+1)<<' '<<rsum/(testid+1)<<' '<<rp_n/(testid+1)<<"\t"<<lsum_filter/(testid+1)<<' '<<lp_n_filter/(testid+1)<<' '<<rsum_filter/(testid+1)<<' '<<rp_n_filter/(testid+1)<<endl;
				cout<<"\t"<<mid_sum/(testid+1)<<' '<<mid_p_n/(testid+1)<<"\t"<<mid_sum_filter/(testid+1)<<' '<<mid_p_n_filter/(testid+1)<<endl;				
			}
		}
		cout<<"relation prediction  MR  MRR  Hits@n:\t"<<mid_sum_filter/(fb_l.size()/2+1)<<'  '<<mrr_midsum_filter/(fb_l.size()/2+1)<<'  '<<mid_p_n_filter/(fb_l.size()/2+1)<<endl;
		cout<<"entity prediction  MR  MRR  Hits@n:\t"<<(lsum_filter+rsum_filter)/fb_l.size()<<'  '<<(mrr_lsum_filter+mrr_rsum_filter)/fb_l.size()<<'  '<<(lp_n_filter+rp_n_filter)/fb_l.size()<<endl;
		cout<<"head entity prediction  Hits@10:\t"<<l_one2one/one2one_num<<"  "<<l_one2n/one2n_num<<" "<<l_n2one/n2one_num<<' '<<l_n2n/n2n_num<<endl;
        cout<<"tail entity prediction  Hits@10:\t"<<r_one2one/one2one_num<<" "<<r_one2n/one2n_num<<" "<<r_n2one/n2one_num<<' '<<r_n2n/n2n_num<<endl;
    }

};
Test test;

void prepare()
{
	cout<<"------------The test process for EngineKG!------------\n";
        FILE* f1 = fopen(("./data/" + data_dir + "/entity2id.txt").c_str(),"r");
	FILE* f2 = fopen(("./data/" + data_dir + "/relation2id.txt").c_str(),"r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
    FILE* f_kb = fopen(("./data/" + data_dir + "/test_pra.txt").c_str(),"r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel;
		fscanf(f_kb,"%d",&rel);
		fscanf(f_kb,"%d",&x);
		vector<pair<vector<int>,double> > b;
		b.clear();
		for (int i = 0; i<x; i++)
		{
			int y,z;
			vector<int> rel_path;
			rel_path.clear();
			fscanf(f_kb,"%d",&y);
			for (int j=0; j<y; j++)
			{
				fscanf(f_kb,"%d",&z);
				rel_path.push_back(z);
			}
			double pr;
			fscanf(f_kb,"%lf",&pr);
			b.push_back(make_pair(rel_path,pr));
		}
		b.clear();
        test.add(e1,e2,rel,b);
    }
    fclose(f_kb);
    FILE* f_path = fopen(("./data/" + data_dir + "/path2.txt").c_str(),"r");
	while (fscanf(f_path,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_path,"%s",buf);
        string s2=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
		fscanf(f_path,"%d",&x);
		vector<pair<vector<int>,double> > b;
		b.clear();
		for (int i = 0; i<x; i++)
		{
			int y,z;
			vector<int> rel_path;
			rel_path.clear();
			fscanf(f_path,"%d",&y);
			for (int j=0; j<y; j++)
			{
				fscanf(f_path,"%d",&z);
				rel_path.push_back(z);
			}
			double pr;
			fscanf(f_path,"%lf",&pr);
			b.push_back(make_pair(rel_path,pr));
		}
        test.add(e1,e2,-1,b);
    }
	fclose(f_path);
    FILE* f_kb1 = fopen(("./data/" + data_dir + "/train.txt").c_str(),"r");
	while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
    FILE* f_kb2 = fopen(("./data/" + data_dir + "/valid.txt").c_str(),"r");
	while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        fscanf(f_kb2,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb2);
	FILE* f_confidence = fopen(("./data/" + data_dir + "/confidence.txt").c_str(),"r");
	while (fscanf(f_confidence,"%d",&x)==1)
	{
		string s = "";
		for (int i=0; i<x; i++)
		{
			fscanf(f_confidence,"%s",buf);
			s = s + string(buf)+" ";
		}
		fscanf(f_confidence,"%d",&x);
		for (int i=0; i<x; i++)
		{
			int y;
			double pr;
			fscanf(f_confidence,"%d%lf",&y,&pr);
			path_confidence[make_pair(s,y)] = pr;
		}
	}
	fclose(f_confidence);
    FILE* f7 = fopen(("./data/" + data_dir + "/n2n.txt").c_str(),"r");
    {
        double n_e1, n_e2;
        while (fscanf(f7,"%lf %lf",&n_e1,&n_e2)==2)
        {
            if (n_e1<1.5)
            {
                if (n_e2<1.5)
                    rel_type.push_back(0);
                else
                    rel_type.push_back(1);

            }
            else
                if (n_e2<1.5)
                    rel_type.push_back(2);
                else
                    rel_type.push_back(3);
        }
    
    fclose(f7);
    }

    int count_rules = 0;
    int rel1, rel2, rel3;
    double confi;
    FILE* f_rulepath = fopen(("./data/" + data_dir + "/Rules/rule_len2.txt").c_str(),"r");
        while (fscanf(f_rulepath,"%d%d", &rel1 ,&rel2)==2)
        {
                fscanf(f_rulepath, "%d%lf", &rel3, &confi);
                rule2rel[make_pair(rel1, rel2)] = make_pair(rel3, confi);
                count_rules++;
        }
        cout<<"The total number of rules in rule_len2.txt is: "<<count_rules<<"\n";

    fclose(f_rulepath);
	
}

int ArgPos(char *str, int argc, char **argv)
{
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a]))
		{
			if (a == argc - 1)
			{
				cout<<"Argument missing for"<<str<<endl;
				exit(1);
			}
			return a;
		}
	return -1;
}

void setparameters(int argc, char **argv)
{
	int i;
	if ((i = ArgPos((char *)"-data_dir", argc, argv)) > 0) data_dir = argv[i + 1];			// storage path of dataset
	if ((i = ArgPos((char *)"-res_path", argc, argv)) > 0) res_path = argv[i + 1];			// storage path of KG embeddings
	if ((i = ArgPos((char *)"-hit_n", argc, argv)) > 0) Hits_N = atoi(argv[i + 1]);			// value of n in Hits@n
}

int main(int argc,char**argv)
{
    setparameters(argc, argv);
	prepare();
    test.run();
    cout<<"Test finish";
}