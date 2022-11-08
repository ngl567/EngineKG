// Train_EngineKG.cpp
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<omp.h>
using namespace std;


#define pi 3.1415926535897932384626433832795


map<vector<int>,string> path2s;  // path convert to string
map<pair<string,int>,double>  path_confidence;

// hyperparameter settings
bool L1_flag=1;
int dimension = 100;
double learning_rate = 0.001;
double margin = 1.0;
double margin_p = 1.0;
double marginrule = 3.0;
int nepoches = 1000;
int nbatches = 100;
string res_path = "/dc_loop";
int loop_n = 0;
int count_rules = 0;
string data_dir = "fb15k";

double th_hc = 0.3;
double th_conf = 0.7;

//normal distribution
double rand(double min, double max)	
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	res = sqrt(res);
	return res;
}


string version;
char buf[100000],buf1[100000],buf2[100000];
int relation_num,entity_num, domain_num;
map<string,int> relation2id, entity2id, domain2id;
map<int,string> id2entity, id2relation, id2domain;
map<pair<int, int>, pair<int, double>> rule2rel;  // 通过规则得到合成的关系		compose paths via rules
map<int, vector<pair<int, double> > > rel2rel;
map<pair<int, int>, int> rule_ok;

map<int, vector<int> > ent2domain;	// entity2domain
map<int, vector<int> > rel2dom_h;	// head concepts of each relation
map<int, vector<int> > rel2dom_t;	// tail concepts of each relation


vector<vector<pair<int,int> > > path;

map<int, map<vector<int>,int> > ok_pre;
map<int, map<pair<int, int>, int > > ok_eval;

map<int, vector<pair<int, int> > > rel2ent;


map<pair<int, int>, int> rule1_ok;
map<int, map<pair<int, int>, int> > rule2_ok;

class Train{

public:
	map<pair<int,int>, map<int,int> > ok;
    void add(int x,int y,int z, vector<pair<vector<int>,double> > path_list)
    {
	// 插入头实体x，尾实体y，关系z，关系路径path_list，ok:表示存在x-z-y的三元组
	// x: head entity, y: tail entity, z: relation z, path_list: path
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
		fb_path.push_back(path_list);
        ok[make_pair(x,z)][y]=1;
    }
    void pop()
    {
        fb_h.pop_back();
		fb_r.pop_back();
		fb_l.pop_back();
		fb_path.pop_back();
    }

	void add_rule(int z, vector<int> rule_body)
    {
	// 插入头实体x，尾实体y，关系z，关系路径path_list，ok:表示存在x-z-y的三元组
	// add rule head z and rule body vector rule_body
        rule_r.push_back(z);
		rule_pre.push_back(rule_body);
        ok_pre[z][rule_body]=1;
    }

	void print_data()
	{
		cout<<"The total number of triples is: "<<fb_h.size()<<"\n";
	}

    // 主函数里调用的运行子程序		KG Embedding training procedure
	void run()
    {
        n = dimension;
        rate = learning_rate;
		regul = 0.01;
		cout<<"n="<<n<<' '<<"rate="<<rate<<endl;
		relation_vec.resize(relation_num);
		for (int i=0; i<relation_vec.size(); i++)
			relation_vec[i].resize(n);
        entity_vec.resize(entity_num);
		for (int i=0; i<entity_vec.size(); i++)
			entity_vec[i].resize(n);
        relation_tmp.resize(relation_num);
		for (int i=0; i<relation_tmp.size(); i++)
			relation_tmp[i].resize(n);
        entity_tmp.resize(entity_num);
		for (int i=0; i<entity_tmp.size(); i++)
			entity_tmp[i].resize(n);
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]);
        }
        train_step();
    }

	// Rule Learning Procedure
	int run_rule()
    {
		n = 100;
		FILE* f_relemb = fopen(("./res/" + res_path + "/relation2vec_3.txt").c_str(), "r");
		FILE* f_entemb = fopen(("./res/" + res_path + "/entity2vec_3.txt").c_str(), "r");
		
		// load the learned embeddings
		relation_vec.resize(relation_num);
		for (int i=0; i<relation_vec.size(); i++)
		{
			relation_vec[i].resize(n);
			for (int ii=0; ii<n; ii++)
				fscanf(f_relemb, "%lf", &relation_vec[i][ii]);
		}
			entity_vec.resize(entity_num);
		for (int i=0; i<entity_vec.size(); i++)
		{
			entity_vec[i].resize(n);
			for (int ii=0; ii<n; ii++)
				fscanf(f_entemb, "%lf", &entity_vec[i][ii]);
		}
		fclose(f_relemb);
		fclose(f_entemb);

        return rule_learning();
    }

private:
    int n;
    double res;//loss function value
    double count,count1;//loss function gradient
    double rate;//learning rate
    double belta;
    double regul; //regulation factor
    int relrules_used;
    vector<int> fb_h,fb_l,fb_r;  // 三元组的头-尾-关系的ID  head entity ID, tail entity ID, relation ID
    vector<vector<pair<vector<int>,double> > >fb_path;   // 从train_pra.txt中得到的关系路径		relation path derived from train_pra.txt
    vector<vector<double> > relation_vec,entity_vec;   // 学习的实体和关系向量		relation and entity embeddings
    vector<vector<double> > relation_tmp,entity_tmp;
    vector<vector<vector<double> > > R, R_tmp;
	vector<vector<int> >rule_pre;   // rule body
	vector<int> rule_r;
	
    double norm(vector<double> &a)
    {
	// 向量归一化  normalization
        double x = vec_len(a);
        if (x>1)
        for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }
    int rand_max(int x)
    {
	// 生成一个[0,x)范围内的整数	generate a integer in the range of [0,x)
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }

    void train_step()
    {
	// 训练过程   training step
        res=0;
		cout<<"epoch number: "<<nepoches<<"\n";
        int batchsize = fb_h.size()/nbatches;
		relation_tmp=relation_vec;
		entity_tmp = entity_vec;
	
        for (int epoch=0; epoch<nepoches; epoch++)
        {
        	res=0;
			int rules_used = 0;
			relrules_used = 0;
			double rules_average = 0;
			double relrules_average = 0;
			for (int batch = 0; batch<nbatches; batch++)
			{
				int e1 = rand_max(entity_num);
         		for (int k=0; k<batchsize; k++)
         		{
					int entity_neg=rand_max(entity_num);    // 随机得到一个实体ID   get an entity ID randomly
					int i=rand_max(fb_h.size());   // 随机得到一个三元组ID   get a triple randomly
					int e1 = fb_h[i], rel = fb_r[i], e2  = fb_l[i];
					int rand_tmp = rand()%100;					
					if (rand_tmp<25)
					{
					// 25%几率换尾实体		reconstruct a negative triple via replacing the tail entity
						while (ok[make_pair(e1,rel)].count(entity_neg)>0)
							entity_neg=rand_max(entity_num);
                        train_kb(e1,e2,rel,e1,entity_neg,rel,margin);
					}
					else
					if (rand_tmp<50)
					{
					// 25%几率换头实体		reconstruct a negative triple via replacing the head entity
						while (ok[make_pair(entity_neg,rel)].count(e2)>0)
							entity_neg=rand_max(entity_num);
                        train_kb(e1,e2,rel,entity_neg,e2,rel,margin);
					}
					else
					{
					// 50%几率换关系		reconstruct a negative triple via replacing the relation entity
						int rel_neg = rand_max(relation_num);   // 随机得到一个关系ID	get a relation ID randomly
						while (ok[make_pair(e1,rel_neg)].count(e2)>0)
							rel_neg = rand_max(relation_num);
                        			train_kb(e1,e2,rel,e1,e2,rel_neg,margin);
					}
					if (fb_path[i].size()>0)
					{
					// 关系路径的训练
						int rel_neg = rand_max(relation_num);  // 随机生成一个对于关系路径的关系ID   get a path randomly
						while (ok[make_pair(e1,rel_neg)].count(e2)>0)
							rel_neg = rand_max(relation_num);
						for (int path_id = 0; path_id<fb_path[i].size(); path_id++)
						{
						// path_id: 表示第几条路径   the i-th path in the path set
							vector<int> rel_path = fb_path[i][path_id].first;  // 当前路径上所有关系   all the relations along the i-th path
							string  s = "";
							if (path2s.count(rel_path)==0)
							{
							    ostringstream oss;
								for (int ii=0; ii<rel_path.size(); ii++)
								{
									oss<<rel_path[ii]<<" ";
								}
							    s=oss.str();//
								path2s[rel_path] = s;
							}
							s = path2s[rel_path];

							double pr = fb_path[i][path_id].second;  // 当前路径的置信度	the confidence of the path
							double pr_path = 0;
							int rel_integ;
							double confi_integ = 0;
							double confi_path = 1;
							if (path_confidence.count(make_pair(s,rel))>0)
								pr_path = path_confidence[make_pair(s,rel)];
							pr_path = 0.99*pr_path + 0.01;   // 路径和关系的置信度		the path confidence associated with the relation r
							if (rel_path.size() > 1){
							    for (int i = 0; i < rel_path.size(); i++){
							        if (rule2rel.count(make_pair(rel_path[i], rel_path[i+1])) > 0){
										rules_used++;  // 当前epoch被使用的规则数	the amount of rules used in this epoch
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
							train_path(rel,rel_neg,rel_path,2*margin_p,pr*pr_path*confi_path);
						}
					}
					norm(relation_tmp[rel]);
            		norm(entity_tmp[e1]);
            		norm(entity_tmp[e2]);
            		norm(entity_tmp[entity_neg]);
					e1 = e2;
			
         		}
	            relation_vec = relation_tmp;
	            entity_vec = entity_tmp;
         	}
            cout<<"epoch:"<<epoch<<' '<<res<<endl;
			
	    	if (epoch>100 && (epoch+1)%100==0){
                int save_n = (epoch+1)/100;
                string serial = to_string(save_n);
                FILE* f2 = fopen(("./res/" + res_path + "/relation2vec_"+serial+".txt").c_str(),"w");
                FILE* f3 = fopen(("./res/" + res_path + "/entity2vec_"+serial+".txt").c_str(),"w");
                for (int i=0; i<relation_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                    fprintf(f2,"\n");
                }
                for (int i=0; i<entity_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                    fprintf(f3,"\n");
                }
                fclose(f2);
                fclose(f3);
                cout<<"Saving the training result succeed!"<<endl;
            }

	    }  // for epoch
    }   // for train_step()
    
	double res1;
	// calculate the score specific to a triple
    double calc_kb(int e1,int e2,int rel)
	{
        double sum=0;
        for (int ii=0; ii<n; ii++)
		{
			double tmp = entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
	        if (L1_flag)
				sum+=fabs(tmp);
			else
				sum+=sqr(tmp);
		}
        return sum;
	}

    // 计算两个关系之间的距离	calculate the score specific to an relation pair
    double calc_rule(int rel, int relpn){

	double sum = 0;
	for (int ii = 0; ii < n; ii++){
		double tmp = relation_vec[rel][ii] - relation_vec[relpn][ii];
		if (L1_flag)
			sum += fabs(tmp);
		else
			sum += sqr(tmp);
	}
        return sum;
    }

	// calculate the gradient specific a triple
    void gradient_kb(int e1,int e2,int rel, double belta)
    {
        for (int ii=0; ii<n; ii++)
        {
            double x = 2*(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel][ii]-=belta*rate*x;
            entity_tmp[e1][ii]-=belta*rate*x;
            entity_tmp[e2][ii]+=belta*rate*x;
        }
    }

    // 计算规则相关的关系梯度  calculate the gradient specific the relation pair association via the length-1 rule
    void gradient_rule(int rel1, int rel2, double belta)
    {
	for (int ii=0; ii<n; ii++){
		double x = 2*(relation_vec[rel1][ii] - relation_vec[rel2][ii]);
		if (L1_flag)
			if (x>0)
				x = 1;
			else
				x = -1;
		relation_tmp[rel1][ii] += belta*rate*x;
		relation_tmp[rel2][ii] -= belta*rate*x;
	}
    }

    // 计算路径关系距离 	calculate the score specific to a path and a relation
	double calc_path(int r1,vector<int> rel_path)
    {
        double sum=0;
        for (int ii=0; ii<n; ii++)
		{
			double tmp = relation_vec[r1][ii];

			for (int j=0; j<rel_path.size(); j++)
				tmp-=relation_vec[rel_path[j]][ii];   // 直接相加路径语义集成	numberical path composition
	        if (L1_flag)
				sum+=fabs(tmp);
			else
				sum+=sqr(tmp);
		}
        return sum;
    }

	// calculate the gradient specific the path via the length-2 rule
    void gradient_path(int r1,vector<int> rel_path, double belta)
    {
        for (int ii=0; ii<n; ii++)
        {

			double x = relation_vec[r1][ii];
			for (int j=0; j<rel_path.size(); j++)
				x-=relation_vec[rel_path[j]][ii];
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[r1][ii]+=belta*rate*x;
			for (int j=0; j<rel_path.size(); j++)
            	relation_tmp[rel_path[j]][ii]-=belta*rate*x;

        }
    }

    // training procedure of the triple and the relation association
	void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,double margin)
    {
        double sum1 = calc_kb(e1_a,e2_a,rel_a);
        double sum2 = calc_kb(e1_b,e2_b,rel_b);
		double lambda_rule = 3;
        if (sum1+margin>sum2)
        {
        	res+=margin+sum1-sum2;
        	gradient_kb(e1_a, e2_a, rel_a, -1);
			gradient_kb(e1_b, e2_b, rel_b, 1);
        }
		if (rel2rel.count(rel_a) > 0)
		{
			for (int i = 0; i < rel2rel[rel_a].size(); i++){
				int rel_rpos = rel2rel[rel_a][i].first;
				double rel_pconfi = rel2rel[rel_a][i].second;
				double sum_pos = calc_rule(rel_a, rel_rpos);
				int rel_rneg = rand_max(relation_num);
				while (rule_ok.count(make_pair(rel_a, rel_rneg)) > 0)
					rel_rneg = rand_max(relation_num);
				double sum_neg = calc_rule(rel_a, rel_rneg);
				if (sum_pos + marginrule > sum_neg){
					res += margin + sum_pos*rel_pconfi - sum_neg;
					gradient_rule(rel_a, rel_rpos, -rel_pconfi*lambda_rule);
					gradient_rule(rel_a, rel_rneg, lambda_rule);
				}
				norm(relation_tmp[rel_a]);
				norm(relation_tmp[rel_rpos]);
				norm(relation_tmp[rel_rneg]);
				relrules_used++;
			}
		}
	}

    // training procedure of the path
	void train_path(int rel, int rel_neg, vector<int> rel_path, double margin, double x)
    {
        double sum1 = calc_path(rel,rel_path);
        double sum2 = calc_path(rel_neg,rel_path);
		double lambda = 1;
        if (sum1+margin>sum2)
        {
        	res+=x*lambda*(margin+sum1-sum2);
        	gradient_path(rel,rel_path, -x*lambda);
			gradient_path(rel_neg,rel_path, x*lambda);
        }
    }


	// rule learning procedure
	int rule_learning()
    {
		FILE* f_rule1 = fopen(("./data/" + data_dir + "/Rules/rule_len1.txt").c_str(), "a+");
		FILE* f_rule2 = fopen(("./data/" + data_dir + "/Rules/rule_len2.txt").c_str(), "a+");
		int rule1_count=0;
		int rule2_count=0;
		
		#pragma omp parallel for
		for (int i=0; i<rule_r.size(); i++)
		{
			int rel = rule_r[i];  // rule head
			vector<int> rule = rule_pre[i];
			int sim_dom;	// global similarity in the latest version
			int rule_n = rule.size();	// the length of the rule
			sim_dom = 0;
			if (rule.size()==1)
			{
				if (rule[0] == rel)	// 规则体和规则头相同跳过	filter out the rules with the same head and body
					continue;

				for (auto & j : rel2dom_h[rel]){
					for (auto & n : rel2dom_h[rule[0]])
						if (j == n){
							sim_dom += 1;
						}
				}

				for (auto & j : rel2dom_t[rel]){
					for (auto & n : rel2dom_t[rule[0]])
						if (j == n){
							sim_dom += 1;
						}
				}
				if (sim_dom == 0)
					continue;
			}
			else
			{
				// co-occurrence similarities specific to the head arguments and the tail arguments
				for (auto & j : rel2dom_h[rel]){
					for (auto & n : rel2dom_h[rule[0]])
						if (j == n){
							sim_dom += 1;
						}
				}

				for (auto & j : rel2dom_t[rel]){
					for (auto & n : rel2dom_t[rule[1]])
						if (j == n){
							sim_dom += 1;
						}
				}
				if (sim_dom == 0)
					continue;
			}
			sim_dom = double(sim_dom);
			sim_dom = sim_dom / domain_num / (2*rule.size()); 

			double score_rule = 0.0;
			for (int jj=0; jj<n; jj++)
			{
				double tmp = relation_vec[rel][jj];

				for (int j=0; j<rule.size(); j++)
					tmp -= relation_vec[rule[j]][jj];   // 直接相加路径语义集成		global similarity via path embedding
				if (L1_flag)
					score_rule += fabs(tmp);
				else
					score_rule += sqr(tmp);
			}
			double score_sum = score_rule - sim_dom;	// overall score

			if (rule.size()==1)
			{
				if (score_sum<=5.0)		// default: 5.0
				{
					double head = rel2ent[rel].size();
					double body = rel2ent[rule[0]].size();
					double support = 0;
					vector<pair<int,int> > supp_ent;
					map<pair<int, int>, int> supp_ok;
					supp_ent.clear();
					supp_ok.clear();
					for (auto & j : rel2ent[rel])
					{
						if (ok_eval[rule[0]].count(j) > 0 && supp_ok.count(j)==0)
						{
							supp_ent.push_back(j);
							supp_ok[j] = 1;
						}								
					}

					double sc = supp_ent.size() / body;
					double hc = supp_ent.size() / head;
					
					if (sc>=th_conf && hc>=th_hc)
					{
						if (rule1_ok.count(make_pair(rel, rule[0]))==0)
						{
							fprintf(f_rule1, "%d\t%d\t%lf\n", rule[0], rel, sc);
							rule1_count++;
							rule1_ok[make_pair(rel, rule[0])] = 1;
						}
					}
				}
			}
			
			else if (rule.size()==2)
			{
				if (score_sum<=5.0)		// 6.5
				{
					double head = rel2ent[rel].size();
					double body = 0;
					double support = 0;

					vector<pair<int,int> > supp_ent;
					map<pair<int, int>, int> supp_ok;
					supp_ent.clear();
					supp_ok.clear();

					vector<pair<int,int> > body_ent;
					map<pair<int, int>, int> body_ok;
					body_ent.clear();
					body_ok.clear();

					for (auto & rb1 : rel2ent[rule[0]]){
						for (auto & rb2 : rel2ent[rule[1]])
						{
							if (rb1.second == rb2.first)
							{
								if (body_ok.count(make_pair(rb1.first, rb2.second))==0)
								{
									body_ent.push_back(make_pair(rb1.first, rb2.second));
									body_ok[make_pair(rb1.first, rb2.second)] = 1;
								}
								if (ok_eval[rel].count(make_pair(rb1.first, rb2.second)) > 0 && supp_ok.count(make_pair(rb1.first, rb2.second))==0)
									supp_ent.push_back(make_pair(rb1.first, rb2.second));
									supp_ok[make_pair(rb1.first, rb2.second)] = 1;
							}
						}
					}

					double sc = supp_ent.size() / body_ent.size();
					double hc = supp_ent.size() / head;

					if (sc >= th_conf && hc >= th_hc)
					{						
						if (rule2_ok[rel].count(make_pair(rule[0], rule[1]))==0)
						{
							fprintf(f_rule2, "%d\t%d\t%d\t%lf\n", rule[0], rule[1], rel, sc);
							rule2_count++;
							rule2_ok[rel][make_pair(rule[0], rule[1])] = 1;
						}						
					}
				}
			}
		}  // for rule_i
		fclose(f_rule1);
		fclose(f_rule2);

		return rule1_count + rule2_count;
    }   // for rule_learning()
};

Train train;
void prepare()
{
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
		id2relation[x+1345] = "-"+st;
		relation_num++;
	}

    FILE* f_domain = fopen(("./data/" + data_dir + "/domain2id.txt").c_str(),"r");
	while (fscanf(f_domain,"%s%d",buf,&x)==2)
	{
		string st=buf;
		domain2id[st]=x;
		id2domain[x]=st;
		domain_num++;
	}

	FILE* f_domh = fopen(("./data/" + data_dir + "/rel2dom_h.txt").c_str(), "r");
    while (fscanf(f_domh, "%s%d", buf, &x) == 2)
    {
	int rel = relation2id[buf];
	vector<int> domain_list;
	domain_list.clear();
	for (int i=0; i<x; i++)
	{
		fscanf(f_domh, "%s", buf);
		int dom = domain2id[buf];
		domain_list.push_back(dom);
	}
	rel2dom_h[rel] = domain_list;
    }
    fclose(f_domh);

    FILE* f_domt = fopen(("./data/" + data_dir + "/rel2dom_t.txt").c_str(), "r");
    while (fscanf(f_domt, "%s%d", buf, &x) == 2)
    {
	int rel = relation2id[buf];
	vector<int> domain_list;
	domain_list.clear();
	for (int i=0; i<x; i++)
	{
		fscanf(f_domt, "%s", buf);
		int dom = domain2id[buf];
		domain_list.push_back(dom);
	}
	rel2dom_t[rel] = domain_list;
    }
    fclose(f_domt);
	

    FILE* fin = fopen(("./data/" + data_dir + "/entity2domain.txt").c_str(), "r");
    int dom, ent;
    while (fscanf(fin, "%s%d", buf, &x) == 2)
    {
	ent = entity2id[buf];
        vector<int> domain_list;
	domain_list.clear();
	for (int i = 0; i < x; i++)
	{
		fscanf(fin, "%s", buf);
		dom = domain2id[buf];
		domain_list.push_back(dom);
	}
	ent2domain[ent] = domain_list;
    }
    fclose(fin);
    cout<<"entity2domain.txt loaded.\n";

    FILE* f_kb = fopen(("./data/" + data_dir + "/train_pra.txt").c_str(),"r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;  // 头实体字符表示	head entity
        fscanf(f_kb,"%s",buf);
        string s2=buf;  // 尾实体字符表示	head entity
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        int e1 = entity2id[s1];  // 关系路径头实体ID	head entity of linked by a path
        int e2 = entity2id[s2];  // 关系路径尾实体ID	tail entity of linked by a path
        int rel;
		fscanf(f_kb,"%d",&rel);  // 读关系rel		relation
		ok_eval[rel][make_pair(e1, e2)] = 1;
		rel2ent[rel].push_back(make_pair(e1, e2));
		fscanf(f_kb,"%d",&x);    // 路径个数		amount of the paths
		vector<pair<vector<int>,double> > b;  // 所有路径的表示		all the path embeddings
		b.clear();
		for (int i = 0; i<x; i++)
		{
			int y,z;
			vector<int> rel_path;
			rel_path.clear();
			fscanf(f_kb,"%d",&y);  // 当前路径长度		length of the path
			for (int j=0; j<y; j++)
			{
				fscanf(f_kb,"%d",&z);  // 读取关系ID	relation ID
				rel_path.push_back(z); // 存入rel_path	store path in the path set rel_path
			}
			double pr;
			fscanf(f_kb,"%lf",&pr);   // 读取路径置信度		get the confidence
			b.push_back(make_pair(rel_path,pr));
			if (ok_pre[rel].count(rel_path) > 0)
				continue;
			else
				train.add_rule(rel, rel_path);
		}
        train.add(e1,e2,rel,b);
    }
	cout<<"train_pra.txt file loaded.\n";
	train.pop();
	relation_num*=2;
	train.print_data();
   
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
    cout<<"domain_num="<<domain_num<<endl;
	
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
    fclose(f_kb);
    cout<<"confidence.txt file loaded.\n";

    cout<<"Loading all the rules......\n";
	string loop_str = to_string(loop_n);
    FILE* f_rule1 = fopen(("./data/" + data_dir + "/Rules/rule_len1.txt").c_str(),"r");
	int rel1, rel2, rel3;
	double confi;
	cout<<"loading length-1 rules!\t"<<"\n";	
	// 读入长度为1的规则	load length-1 rules
	while (fscanf(f_rule1,"%d", &rel1)==1)
	{
		fscanf(f_rule1, "%d%lf", &rel2, &confi);
		rel2rel[rel1].push_back(make_pair(rel2, confi));
		rule_ok[make_pair(rel1, rel2)] = 1;
		count_rules++;
    }
    fclose(f_rule1);

    FILE* f_rule2 = fopen(("./data/" + data_dir + "/Rules/rule_len2.txt").c_str(),"r");
	cout<<"loading length-2 rules!\t"<<"\n";	
	// 读入长度为2的规则	load length-2 rules
	while (fscanf(f_rule2,"%d%d", &rel1 ,&rel2)==2)
	{
		fscanf(f_rule2, "%d%lf", &rel3, &confi);
		rule2rel[make_pair(rel1, rel2)] = make_pair(rel3, confi);
		count_rules++;
	}
	cout<<"The confidence of rules is: 0.7"<<"\n";
	cout<<"The total number of seed rules is: "<<count_rules<<"\n";
    fclose(f_rule2);
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
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);		// embedding dimension
	if ((i = ArgPos((char *)"-lr", argc, argv)) > 0) learning_rate = atof(argv[i + 1]);		// learning rate
	if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) nepoches = atof(argv[i + 1]);		// total epoches
	if ((i = ArgPos((char *)"-nbatch", argc, argv)) > 0) nbatches = atof(argv[i + 1]);		// number of batches
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);		// gamma_1
	if ((i = ArgPos((char *)"-margin_p", argc, argv)) > 0) margin_p = atof(argv[i + 1]);	// gamma_2
	if ((i = ArgPos((char *)"-margin_r", argc, argv)) > 0) marginrule = atof(argv[i + 1]);	// gamma_3
	if ((i = ArgPos((char *)"-data_dir", argc, argv)) > 0) data_dir = argv[i + 1];			// storage path of dataset
	if ((i = ArgPos((char *)"-res_path", argc, argv)) > 0) res_path = argv[i + 1];			// storage path of embeddings
}

int main(int argc,char**argv)
{
	setparameters(argc, argv);
	cout << "Start to prepare!\n";
    prepare();
	cout << "Prepare Success!\n\n";
	cout<<"Hyperparameters:\nmargin=\t"<<margin<<"\nmargin_path=\t"<<margin_p<<"\nmargin_rulerel=\t"<<marginrule<<"learning rate=\t"<<learning_rate<<endl;
	cout<<"amount of batches: "<<nbatches<<"\n\n";
	cout <<"================================\n";
	cout << "Start Training!\n";
	cout <<"================================\n";
	loop_n = 0;
	int rule_new_n = count_rules;
	while(rule_new_n){
		train.run();
		string loop_str = to_string(loop_n);
		cout<<"Training of loop" + loop_str + "has finished!\n";
		rule_new_n = train.run_rule();
		cout<<"iteration "<<loop_str + " KG Embedding and Rule Learning finished\n";
		string rule_new = to_string(rule_new_n);
		cout<<rule_new + "new rules have been mined\n";
	}
}