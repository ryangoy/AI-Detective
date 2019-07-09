import React from "react"
import Spinner from "./Spinner"
// import StageSummary from "./StageSummary"
// import ReactCSSTransitionGroup from 'react-addons-css-transition-group';

export default class Pipeline extends React.Component {
    

    // getListOfItems() {
    //     var items = []
    //     for (var i=0; i<this.props.completed_stages.length; i++) {
    //             items.push(<StageSummary stage={this.props.completed_stages[i]} id={i} />)
    //         }
    //     return items
    // }
    
    // renderItems() {
    //     return (
    //         <div>
    //             <ReactCSSTransitionGroup transitionName="example">
    //                 {this.getListOfItems()}
    //             </ReactCSSTransitionGroup>
    //         </div>
    //     )
    // }

    renderSpinner(){
        return (
            <div className={this.props.show_spinner?'fadeInSpinner':'fadeOutSpinner'}>
                <Spinner stage={this.props.stage}/>
            </div>
        )
    }

    renderResult(){
        return (
            <div className={this.props.show_result?'fadeInText':'fadeOutText'}>
                <h3 className="form-group col-md-5 center-div mt-3">
                    The verdict is in. You're lying with <b>{this.props.percent.toFixed(2)}%</b> confidence, 
                    or telling the truth with <b>{(100-this.props.percent).toFixed(2)}%</b> confidence.
                </h3>
            </div>
        )
    }

    render() {
        if(this.props.stage === 'none') {
            return <h3 className="form-group col-md-5 center-div mt-3">Upload a file to get started!</h3>
        } else{
            return (<div>
                      {this.renderSpinner()}
                      {this.renderResult()}

                    </div>)
        }
    }
}
// {this.renderItems()}
// this.props.completed_stages.map((stage) => {
//                         return (<StageSummary stage={stage} />)
//                     })

// 

//           <div className={this.state.show_spinner?'fadeInSpinner':'fadeOutSpinner'}>
          
//             <img className="row" src={process.env.PUBLIC_URL + "spinner.gif"}/>
//             <p>Running <b>{this.state.stage}</b>...</p>
//           </div>
