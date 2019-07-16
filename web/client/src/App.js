import React, { Component } from "react";
import axios from 'axios';
import { Progress } from 'reactstrap';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css'
import Pipeline from "./components/Pipeline"


// const endpoint = 'https://nwmh21ywva.execute-api.us-west-1.amazonaws.com/dev1'
const endpoint = 'http://0.0.0.0:8000'

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
      stage: 'initialization'
    }
  }

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

  onClickHandler = () => {
    this.setState({show_spinner: true})
    this.onClickHandlerAsync(this.state.selectedFile)
  }

  setStateAsync(state) {
    return new Promise((resolve) => {
      this.setState(state, resolve)
    })
  }

  async onClickHandlerAsync(file) {

    try {
      let presigned_post = await axios.get(endpoint+"/get-presigned-post/" + this.state.fileField)
      let options = {headers: {'Content-Type': 'mp4'}}

      await axios.put(presigned_post.data.url, file, options)
      await axios.get(endpoint+"/predict/" + this.state.fileField, {timeout: 9000000})
      
      this.timer = setInterval(() => this.pollStage(this.state.fileField), 2000)
    } catch(error) {
      console.log('Request to server errored out. Not starting polling process.')
      this.setStateAsync({show_spinner: false})
    }
    return true
  }

  async pollStage(filename) {
    try{
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
    } catch(error) {
      console.log('Cannot poll from server. Stopping polling.')
      clearInterval(this.timer)
      this.timer = null
      this.setStateAsync({show_spinner: false})
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
              <button type="button" className="input-group-append btn btn-primary btn-block" onClick={this.onClickHandler}>Upload</button>
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
                  show_spinner={this.state.show_spinner}
                  stage={this.state.stage} 
                  show_result={this.state.show_result}
                  percent={this.state.percent}/>
        </div>
          
          

      </div>
    )
  }
}

export default App;
