import React, { Component } from "react";
import axios from 'axios';

import { Progress } from 'reactstrap';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css'
import Pipeline from "./components/Pipeline"


// import openSocket from 'socket.io-client'

const endpoint = 'https://nwmh21ywva.execute-api.us-west-1.amazonaws.com/dev1'
// const endpoint = 'http://0.0.0.0:8000'

const vidpoint = endpoint + '/get-video'

// const socket = openSocket('http://0.0.0.0:8000');

class App extends Component {

  constructor(props) {
    super(props);
    this.state = {
      selectedFile: null,
      fileField: 'Upload File Here',
      loaded:0,
      percent:-1,
      show_file_upload_progress: false,
      show_spinner: false,
      show_result: false,
      test: 'not set',
      url: null,
      stage: 'none'
      // completed_stages: []
    }
    // this.subscribeToPipelineUpdates();
  }


  // subscribeToPipelineUpdates = () => {
  //   socket.on('stage', (s) => {
        

  //       if (this.state.stage !== 'none') {
  //           this.setState({completed_stages:this.state.completed_stages.concat([this.state.stage])})
  //       } else {
  //           this.setState({show_spinner:true})
  //       }
  //       this.setState({stage:s})
  //       if (s === 'completed') {
  //           this.setState({show_result:true})
  //       }

  //   });
  // } 

  // when a file is selected
  onChangeHandler = (event) =>{
    if(this.checkFileType(event)){ 
      this.setState({
        selectedFile: event.target.files[0],
        fileField: event.target.files[0].name
      })
    }
    return true
  }

  testHandler = () => {
    axios.get(endpoint+"/test").then(res=> {this.setState({test: res.data})})
    return true
  }

  getVideoFile = () => {
    // const vfile = axios.get(endpoint + "/get_video/" + this.state.fileField)
    // return vfile
    return vidpoint + '/' + this.state.fileField
  }


  setStateAsync(state) {
    return new Promise((resolve) => {
      this.setState(state, resolve)
    })
  }

  async onClickHandler(file) {

    await this.setStateAsync({//show_file_upload_progress: true,
                   show_spinner: true
                   // stage: 'none',
                   // completed_stages: []
                 })

    let presigned_post = await axios.get(endpoint+"/get-presigned-post/" + this.state.fileField)
    let options = { 
              headers: {'Content-Type': 'mp4'},
              // onUploadProgress: ProgressEvent => {
              //   this.setState({
              //     loaded: (ProgressEvent.loaded / ProgressEvent.total*100),
              //   })
              // }
            }

    await axios.put(presigned_post.data.url, file, options)
    try {
      axios.get(endpoint+"/predict/" + this.state.fileField, {timeout: 9000000})
    }
    catch(error) {
      if (error.response && error.response.status === 504) {
        console.log("30 second API Gateway timeout. Function continuing to run.");
      } else {
        console.log(error)
      } 
    }


    this.timer = setInterval(() => this.pollStage(this.state.fileField), 1000)

    return true
  }


  async pollStage(filename) {
    let response = await axios.get(endpoint+"/poll-stage/"+filename)
    let percent = response.data['prediction']
    let stage = response.data['stage']
    if (stage === 'finished') {
      await this.setStateAsync({percent:percent, show_spinner:false, show_result:true})
      clearInterval(this.timer)
      this.timer = null
    } else{
      await this.setStateAsync({stage:stage})
    }
    
  }

  // when we press the upload button
  onClickHandler_old = () => {

    if (this.state.selectedFile == null) {
      console.log('File is null')
    }
    else {
      this.setState({show_file_upload_progress: true,
                     show_spinner: true
                     // stage: 'none',
                     // completed_stages: []
                    })
      const data = new FormData() 
      data.append('file', this.state.selectedFile)

      axios.post(endpoint + "/predict", data, {
        
        onUploadProgress: ProgressEvent => {
          this.setState({
               loaded: (ProgressEvent.loaded / ProgressEvent.total*100),
          })
        }

      }).then(res => {
        const percent = res.data['percent'];
        return percent
      }).then(percent => {
        this.setState({percent:percent, show_spinner:false, show_result:true});
      }).catch(error => {
        console.log(error.response)
      })

    }
  }

  // ensure it is an image
  checkFileType= (event) =>{

    let files = event.target.files[0]
    let err = ''
    const types = ['video/mp4']

    if (types.every(type => files.type !== type)) {
      err = files.type+' is not a supported format\n';
    }

    if (err !== '') {
        event.target.value = null
        toast(err)
        return false; 
    }
    return true;

  }


  render() { return (
      <div>
        <form action="/upload" method="POST" encType="multipart/form-data">
          <div className="input-group col-md-5 center-div">
            <div className="custom-file">
              <input type="file" className="custom-file-input" id="inputGroupFile02" onChange={this.onChangeHandler} 
                aria-describedby="inputGroupFileAddon01"/>
              <label className="custom-file-label">{this.state.fileField}</label>
            </div>
            <div className="input-group-append">
              <button type="button" className="input-group-append btn btn-primary btn-block" onClick={() => this.onClickHandler(this.state.selectedFile)}>Upload</button>
            </div>
          </div>
        </form>
        <div className="form-group">
         <ToastContainer />
        </div>

        { this.state.show_file_upload_progress ?
          <div className="form-group col-md-5 center-div mt-3">
              <Progress max="100" color="success" value={this.state.loaded} >{Math.round(this.state.loaded,2) }%</Progress>
          </div>
          :
          <div></div>
        }
        <div>
        <Pipeline 
                  // stage={this.state.stage} 
                  // completed_stages={this.state.completed_stages}
                  show_spinner={this.state.show_spinner}
                  show_result={this.state.show_result}
                  percent={this.state.percent}/>
        </div>
          
          

      </div>
    )
  }
}

export default App;
