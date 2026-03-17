classdef PhaseScheduler < handle
% PhaseScheduler - 深度融合调度器
% 实现阶段性交替进化策略：
% EVOLUTION 阶段 (3代) -> DIFFUSION 阶段 (1代) -> 更新 DM -> 循环
%
% 作者: AI Assistant
% 日期: 2026-03-11

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
            % 构造函数
            % phase_length: 进化阶段长度（建议 3）
            
            obj.PhaseLength = phase_length;
            obj.PhaseCounter = 0;
            obj.CurrentPhase = 'EVOLUTION';
            obj.GenerationBuffer = [];
            obj.DMTriggerCount = 0;
            obj.ArchiveQuality = 0.5;
        end
        
        function [use_dm, should_collect, should_update_dm] = getStrategy(obj, generation, dm_success_rate)
            % 获取当前策略
            % 
            % 输入:
            %   generation - 当前代数
            %   dm_success_rate - DM 最近成功率
            %
            % 输出:
            %   use_dm - 是否使用扩散模型
            %   should_collect - 是否收集解到 buffer
            %   should_update_dm - 是否更新 DM 模型
            
            obj.PhaseCounter = obj.PhaseCounter + 1;
            should_collect = false;
            should_update_dm = false;
            
            switch obj.CurrentPhase
                case 'EVOLUTION'
                    % 进化阶段：纯 GA
                    use_dm = false;
                    should_collect = true;  % 收集解
                    
                    if obj.PhaseCounter >= obj.PhaseLength
                        % 进化阶段结束，准备切换到扩散阶段
                        obj.CurrentPhase = 'DIFFUSION';
                        obj.PhaseCounter = 0;
                        use_dm = true;
                        should_collect = false;
                        
                        % 评估是否更新 DM
                        if dm_success_rate < 0.15 || obj.DMTriggerCount >= 2
                            should_update_dm = true;
                            obj.DMTriggerCount = 0;
                        else
                            obj.DMTriggerCount = obj.DMTriggerCount + 1;
                        end
                    end
                    
                case 'DIFFUSION'
                    % 扩散阶段：使用 DM
                    use_dm = true;
                    should_collect = true;
                    
                    % 扩散只进行 1 代
                    obj.CurrentPhase = 'EVOLUTION';
                    obj.PhaseCounter = 0;
            end
        end
        
        function addToBuffer(obj, Solutions)
            % 添加解到 buffer
            if isempty(obj.GenerationBuffer)
                obj.GenerationBuffer = Solutions;
            else
                obj.GenerationBuffer = [obj.GenerationBuffer, Solutions];
            end
            
            % 限制 buffer 大小
            if length(obj.GenerationBuffer) > 500
                obj.GenerationBuffer = obj.GenerationBuffer(end-499:end);
            end
        end
        
        function Buffer = getBuffer(obj)
            % 获取 buffer 并清空
            Buffer = obj.GenerationBuffer;
            obj.GenerationBuffer = [];
        end
        
        function reset(obj)
            % 重置调度器
            obj.PhaseCounter = 0;
            obj.CurrentPhase = 'EVOLUTION';
            obj.GenerationBuffer = [];
            obj.DMTriggerCount = 0;
        end
        
        function phase = getCurrentPhase(obj)
            % 获取当前阶段
            phase = obj.CurrentPhase;
        end
    end
end
