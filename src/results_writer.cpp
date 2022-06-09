#include <iostream>
#include <fstream>
#include "../include/results_writer.h"


ResultsWriter::ResultsWriter( const std::string &dataset_name, 
                              const std::string &results_folder ) : 
  results_file_(results_folder + "/"  + dataset_name + "_results.txt"){}

bool ResultsWriter::addResults( const std::string &test_image_name, 
                                const std::string &match_template_name, 
                                int pos_x, int pos_y )
{
  auto iter = results_.find(test_image_name);
  if( iter != results_.end() )
  {
    if(iter->second.size() >= 10 )
    {
      std::cout<<"Error: Only the first ten best matches are required, discarding the current result"<<std::endl;
      return false;
    }
  }
  results_[test_image_name].push_back(ResItem(match_template_name, pos_x, pos_y));
  return true;
}

bool ResultsWriter::write()
{
  if(results_.size() != 10 )
  {
    std::cout<<"Can't write the results: The algorithm must be tested on"<<std::endl;
    std::cout<<"exactly 10 images (test0.jpg, ..., test9.jpg) for each dataset"<<std::endl;
    return false;
  }
  
  for( auto &p : results_ )
  {
    if( p.second.size() != 10 )
    {
      std::cout<<"Can't write the results: For each test image"<<std::endl;
      std::cout<<"exactly ten best matches are required"<<std::endl;
      return false;
    }
  }

  std::ofstream out_file;
  out_file.open(results_file_);
  
  
  for( auto &p : results_ )
  {
    out_file<<p.first<<" ";

    for( auto &res : p.second )
    {
      out_file<<res.template_name<<" "<<res.x<<" "<<res.y<<" ";
    }
    out_file<<std::endl;
  }
  
  out_file.close();
  
  return true;
}

