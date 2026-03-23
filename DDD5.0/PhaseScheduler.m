classdef PhaseScheduler < handle
% PhaseScheduler - 深度融合调度器
% 实现阶段性交替进化策略：
% EVOLUTION 阶段 (3代) -> DIFFUSION 阶段 (1代) -> 更新 DM -> 循环
%
% 作者: AI Assistant
% 日期: 2026-03-11
%未被调用

    properties
        PhaseLength       % 每个进化阶段的长度（代）
        PhaseCounter      % 当前阶段计数
        CurrentPhase      % 'EVOLUTION' 或 'DIFFUSION'
        GenerationBuffer  % 阶段性收集的解
        DMTriggerCount    % DM 触发计数
        ArchiveQuality    % 档案质量评估
    end
    
    methods
        function obj = PhaseScheduler(phase_length)
            obj.PhaseLength = phase_length;
            obj.PhaseCounter = 0;
            obj.CurrentPhase = 'EVOLUTION';
            obj.GenerationBuffer = [];
            obj.DMTriggerCount = 0;
            obj.ArchiveQuality = 0.5;
        end
        
        function [use_dm, should_collect, should_update_dm] = getStrategy(obj, generation, dm_success_rate)
            obj.PhaseCounter = obj.PhaseCounter + 1;
            should_collect = false;
            should_update_dm = false;
            
            switch obj.CurrentPhase
                case 'EVOLUTION'
                    use_dm = false;
                    should_collect = true;
                    
                    if obj.PhaseCounter >= obj.PhaseLength
                        obj.CurrentPhase = 'DIFFUSION';
                        obj.PhaseCounter = 0;
                        use_dm = true;
                        should_collect = false;
                        
                        if dm_success_rate < 0.15 || obj.DMTriggerCount >= 2
                            should_update_dm = true;
                            obj.DMTriggerCount = 0;
                        else
                            obj.DMTriggerCount = obj.DMTriggerCount + 1;
                        end
                    end
                    
                case 'DIFFUSION'
                    use_dm = true;
                    should_collect = true;
                    
                    obj.CurrentPhase = 'EVOLUTION';
                    obj.PhaseCounter = 0;
            end
        end
        
        function addToBuffer(obj, Solutions)
            if isempty(obj.GenerationBuffer)
                obj.GenerationBuffer = Solutions;
            else
                obj.GenerationBuffer = [obj.GenerationBuffer, Solutions];
            end
            
            if length(obj.GenerationBuffer) > 500
                obj.GenerationBuffer = obj.GenerationBuffer(end-499:end);
            end
        end
        
        function Buffer = getBuffer(obj)
            Buffer = obj.GenerationBuffer;
            obj.GenerationBuffer = [];
        end
        
        function reset(obj)
            obj.PhaseCounter = 0;
            obj.CurrentPhase = 'EVOLUTION';
            obj.GenerationBuffer = [];
            obj.DMTriggerCount = 0;
        end
        
        function phase = getCurrentPhase(obj)
            phase = obj.CurrentPhase;
        end
    end
end
