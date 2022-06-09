#pragma once

#include <string>
#include <vector>
#include <map>

class ResultsWriter
{  
public:
  
  ResultsWriter( const std::string &dataset_name, 
                 const std::string &results_folder =  "."  );
  bool addResults( const std::string &test_image_name, 
                   const std::string &match_template_name, 
                   int pos_x, int pos_y );
  bool write();

private:
  
  struct ResItem
  {
    ResItem(const std::string &template_name, int x, int y ) :
      template_name(template_name), x(x), y(y){};
    std::string template_name;
    int x, y;
  };
  
  std::map< std::string, std::vector<ResItem> > results_;
  std::string results_file_;
};
